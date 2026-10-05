package io.github.capsopasme.mossnano

import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/** Resumable downloader for the model files (foreground, survives screen-off). */
class ModelDownloadService : Service() {

    data class Progress(
        val running: Boolean = false,
        val variant: ModelVariant? = null,
        val file: String = "",
        val done: Long = 0,
        val total: Long = 0,
        val bytesPerSec: Long = 0,
        val error: String? = null,
        val finished: Boolean = false,
    )

    companion object {
        private const val TAG = "ModelDownload"
        const val ACTION_START = "start"
        const val ACTION_CANCEL = "cancel"
        const val EXTRA_VARIANT = "variant"
        const val EXTRA_WITH_CLONE = "withClone"

        private val _progress = MutableStateFlow(Progress())
        val progress: StateFlow<Progress> = _progress

        fun start(context: Context, variant: ModelVariant, withClone: Boolean) {
            context.startForegroundService(
                Intent(context, ModelDownloadService::class.java).setAction(ACTION_START)
                    .putExtra(EXTRA_VARIANT, variant.name).putExtra(EXTRA_WITH_CLONE, withClone)
            )
        }

        fun cancel(context: Context) {
            context.startService(Intent(context, ModelDownloadService::class.java).setAction(ACTION_CANCEL))
        }
    }

    @Volatile private var cancelled = false
    private var worker: Thread? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                foreground("准备下载…", -1)
                if (worker?.isAlive == true) return START_NOT_STICKY
                val variant = ModelVariant.valueOf(intent.getStringExtra(EXTRA_VARIANT) ?: ModelVariant.FP32.name)
                val withClone = intent.getBooleanExtra(EXTRA_WITH_CLONE, true)
                cancelled = false
                worker = thread(name = "moss-download") { run(variant, withClone) }
            }
            ACTION_CANCEL -> {
                cancelled = true
                if (worker?.isAlive != true) stopSelfNow()
            }
        }
        return START_NOT_STICKY
    }

    private fun foreground(text: String, permille: Int) {
        val n = Notifications.downloading(this, text, permille)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(Notifications.ID_DOWNLOAD, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(Notifications.ID_DOWNLOAD, n)
        }
    }

    private fun run(variant: ModelVariant, withClone: Boolean) {
        val wl = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mossnano:download")
        wl.acquire(3 * 60 * 60 * 1000L)
        val source = AppSettings(this).downloadSource
        val files = ModelStore.files(variant).filter { withClone || !it.cloneOnly }
        val knownTotal = files.sumOf { maxOf(it.size, 0L) }
        var lastUi = 0L
        var lastBytes = 0L
        var lastTime = System.currentTimeMillis()
        var speed = 0L
        try {
            // A model switch must not leave a half-loaded engine around.
            EngineManager.release()
            var doneBefore = 0L
            for (f in files) {
                if (cancelled) break
                val target = ModelStore.file(this, variant, f)
                if (ModelStore.isPresent(this, variant, f)) {
                    doneBefore += target.length()
                    continue
                }
                val url = ModelStore.url(source, variant, f)
                Log.i(TAG, "GET $url")
                download(url, target, f.size) { fileDone, fileTotal ->
                    val now = System.currentTimeMillis()
                    if (now - lastUi > 400) {
                        val total = if (knownTotal > 0) knownTotal else doneBefore + fileTotal
                        val done = doneBefore + fileDone
                        if (now - lastTime > 1000) {
                            speed = (done - lastBytes) * 1000 / (now - lastTime)
                            lastBytes = done
                            lastTime = now
                        }
                        _progress.value = Progress(true, variant, f.name, done, total, speed)
                        val permille = if (total > 0) (done * 1000 / total).toInt() else -1
                        getSystemService(NotificationManager::class.java).notify(
                            Notifications.ID_DOWNLOAD,
                            Notifications.downloading(this, "${f.name} · ${done / 1_048_576}/${total / 1_048_576} MB", permille),
                        )
                        lastUi = now
                    }
                    !cancelled
                }
                doneBefore += target.length()
            }
            _progress.value = if (cancelled) Progress(error = "已取消", variant = variant)
            else Progress(finished = true, variant = variant, done = doneBefore, total = doneBefore)
        } catch (t: Throwable) {
            Log.e(TAG, "download failed", t)
            _progress.value = Progress(variant = variant, error = t.message ?: t.toString())
        } finally {
            if (wl.isHeld) wl.release()
            stopSelfNow()
        }
    }

    /**
     * Downloads [url] to [target] via a ".part" file with HTTP Range resume.
     * Redirects are followed manually (HF -> CDN) so the Range header is kept.
     */
    private fun download(url: String, target: File, expectedSize: Long, onProgress: (Long, Long) -> Boolean) {
        target.parentFile?.mkdirs()
        val part = File(target.path + ".part")
        var attempt = 0
        while (true) {
            try {
                downloadOnce(url, part, onProgress)
                break
            } catch (e: IOException) {
                if (cancelled || ++attempt >= 5) throw e
                Log.w(TAG, "retry $attempt for $url: ${e.message}")
                Thread.sleep(1500L * attempt)
            }
        }
        if (cancelled) return
        if (expectedSize > 0 && part.length() != expectedSize) {
            part.delete()
            throw IOException("${target.name}: size ${part.length()} != expected $expectedSize")
        }
        target.delete()
        if (!part.renameTo(target)) throw IOException("rename failed: ${target.path}")
    }

    /** Opens [startUrl], following redirects manually. Returns null when the server says the file is complete (416). */
    private fun open(startUrl: String, have: Long): HttpURLConnection? {
        var url = startUrl
        repeat(10) {
            val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "MossTtsNano-Android/1.0")
                if (have > 0) setRequestProperty("Range", "bytes=$have-")
            }
            val code = conn.responseCode
            when {
                code in 300..399 -> {
                    val loc = conn.getHeaderField("Location") ?: throw IOException("redirect without Location")
                    conn.disconnect()
                    url = URL(URL(url), loc).toString()
                }
                code == 416 -> {
                    conn.disconnect()
                    return null
                }
                code == 200 || code == 206 -> return conn
                else -> {
                    conn.disconnect()
                    throw IOException("HTTP $code for $url")
                }
            }
        }
        throw IOException("too many redirects: $startUrl")
    }

    private fun downloadOnce(startUrl: String, part: File, onProgress: (Long, Long) -> Boolean) {
        val have = if (part.exists()) part.length() else 0L
        val conn = open(startUrl, have) ?: return
        try {
            val resume = conn.responseCode == 206
            val length = conn.contentLengthLong
            val total = if (resume) have + length else length
            RandomAccessFile(part, "rw").use { raf ->
                if (resume) raf.seek(have) else raf.setLength(0)
                var done = if (resume) have else 0L
                conn.inputStream.use { input ->
                    val buf = ByteArray(256 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        raf.write(buf, 0, n)
                        done += n
                        if (!onProgress(done, total)) return
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun stopSelfNow() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }
}
