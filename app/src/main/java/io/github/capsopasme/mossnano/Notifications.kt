package io.github.capsopasme.mossnano

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent

object Notifications {
    const val CH_SPEAK = "speak"
    const val CH_DOWNLOAD = "download"
    const val ID_SPEAK = 1
    const val ID_DOWNLOAD = 2

    fun createChannels(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CH_SPEAK, "朗读", NotificationManager.IMPORTANCE_LOW))
        nm.createNotificationChannel(NotificationChannel(CH_DOWNLOAD, "模型下载", NotificationManager.IMPORTANCE_LOW))
    }

    private fun openApp(context: Context): PendingIntent =
        PendingIntent.getActivity(
            context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )

    fun speaking(context: Context, text: String): Notification {
        val stop = PendingIntent.getService(
            context, 1, Intent(context, SpeakService::class.java).setAction(SpeakService.ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(context, CH_SPEAK)
            .setSmallIcon(R.drawable.ic_stat_speak)
            .setContentTitle("MOSS-TTS-Nano 朗读中")
            .setContentText(text.take(80))
            .setContentIntent(openApp(context))
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "停止", stop).build())
            .build()
    }

    fun downloading(context: Context, text: String, progress: Int): Notification {
        val cancel = PendingIntent.getService(
            context, 2, Intent(context, ModelDownloadService::class.java).setAction(ModelDownloadService.ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(context, CH_DOWNLOAD)
            .setSmallIcon(R.drawable.ic_stat_speak)
            .setContentTitle("下载模型")
            .setContentText(text)
            .setProgress(1000, progress, progress < 0)
            .setContentIntent(openApp(context))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(null, "取消", cancel).build())
            .build()
    }
}
