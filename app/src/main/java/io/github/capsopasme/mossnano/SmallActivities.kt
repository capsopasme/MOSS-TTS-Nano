package io.github.capsopasme.mossnano

import android.app.Activity
import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.os.Process
import android.speech.tts.TextToSpeech
import android.widget.Toast
import io.github.capsopasme.mossnano.engine.MossTtsEngine

class MossApp : Application() {
    override fun onCreate() {
        super.onCreate()
        EngineManager.init(this)
        Notifications.createChannels(this)
        MossTtsEngine.threadPriorityHook = { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) }
    }
}

/** Required by the TTS framework: reports which languages are installed. */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ready = ModelStore.isReady(this, AppSettings(this).variant)
        val available = if (ready) arrayListOf("zho-CHN", "eng-USA", "jpn-JPN", "kor-KOR") else arrayListOf()
        val unavailable = if (ready) arrayListOf() else arrayListOf("zho-CHN", "eng-USA")
        setResult(
            if (ready) TextToSpeech.Engine.CHECK_VOICE_DATA_PASS else TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL,
            Intent()
                .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, available)
                .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, unavailable),
        )
        finish()
    }
}

/** "朗读" entry in the text-selection menu (PROCESS_TEXT) and share target (SEND text/plain). */
class ReadAloudActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val text = when (intent?.action) {
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
            else -> null
        }
        if (text.isNullOrBlank()) {
            finish()
            return
        }
        if (!ModelStore.isReady(this, AppSettings(this).variant)) {
            Toast.makeText(this, "请先在 MOSS-TTS-Nano 里下载模型", Toast.LENGTH_LONG).show()
            startActivity(Intent(this, MainActivity::class.java))
        } else {
            SpeakService.speak(this, text)
        }
        finish()
    }
}

/** Settings entry shown in system TTS settings -> opens the main screen. */
class TtsSettingsActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}
