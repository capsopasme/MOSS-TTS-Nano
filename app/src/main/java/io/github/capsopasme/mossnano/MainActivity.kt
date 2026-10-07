package io.github.capsopasme.mossnano

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.capsopasme.mossnano.engine.CpuAffinity
import io.github.capsopasme.mossnano.engine.VoicePrompt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            val dark = isSystemInDarkTheme()
            val scheme = when {
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
                    if (dark) dynamicDarkColorScheme(this) else dynamicLightColorScheme(this)
                dark -> darkColorScheme()
                else -> lightColorScheme()
            }
            MaterialTheme(colorScheme = scheme) {
                Surface(Modifier.fillMaxSize()) { MainScreen() }
            }
        }
    }
}

@Composable
private fun MainScreen() {
    val context = LocalContext.current
    val settings = remember { AppSettings(context) }
    var refresh by remember { mutableIntStateOf(0) }

    val notifLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    LaunchedEffect(Unit) {
        if (Build.VERSION.SDK_INT >= 33 &&
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) notifLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    Column(
        Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Header()
        ModelCard(settings, refresh) { refresh++ }
        SpeakCard(settings, refresh)
        CloneCard(settings, refresh) { refresh++ }
        PerfCard(settings)
        SystemTtsCard()
        BackgroundCard()
        Spacer(Modifier.padding(8.dp))
    }
}

@Composable
private fun Header() {
    val state by EngineManager.state.collectAsStateWithLifecycle()
    Column {
        Text("MOSS-TTS-Nano", fontSize = 26.sp, fontWeight = FontWeight.Bold)
        val s = when (val st = state) {
            EngineManager.State.NotLoaded -> "引擎未加载（首次朗读时自动加载）"
            EngineManager.State.Loading -> "正在加载模型…"
            is EngineManager.State.Ready -> "引擎就绪 · ${st.variant.name} · 加载 ${st.loadMs}ms · 预热 ${st.warmupMs}ms · 本进程原生内存 ${Debug.getNativeHeapAllocatedSize() / 1_048_576}MB"
            is EngineManager.State.Error -> "引擎错误：${st.message}"
        }
        Text(s, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

// --------------------------------------------------------------------------------- model
@Composable
private fun ModelCard(settings: AppSettings, refresh: Int, onChanged: () -> Unit) {
    val context = LocalContext.current
    val progress by ModelDownloadService.progress.collectAsStateWithLifecycle()
    var variant by remember { mutableStateOf(settings.variant) }
    var source by remember { mutableStateOf(settings.downloadSource) }
    var withClone by remember { mutableStateOf(true) }
    val ready = remember(variant, refresh, progress.finished, progress.running) { ModelStore.isReady(context, variant) }
    val cloneReady = remember(variant, refresh, progress.finished, progress.running) { ModelStore.canClone(context, variant) }

    LaunchedEffect(progress.finished) { if (progress.finished) onChanged() }

    SectionCard("模型") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModelVariant.entries.forEach { v ->
                FilterChip(
                    selected = variant == v,
                    onClick = {
                        variant = v
                        settings.variant = v
                        EngineManager.releaseAsync()
                        onChanged()
                    },
                    label = { Text(v.name) },
                )
            }
        }
        Text(variant.label, style = MaterialTheme.typography.bodySmall)
        Text(
            if (variant == ModelVariant.INT8) {
                "推荐。由本仓库 GitHub Actions 从官方 ONNX 量化：LM 全部矩阵权重 int8（走 ARM 点积 / i8mm 整数核），每帧读取的权重约为 FP32 的 1/4，生成更快、更省电；codec 保持官方 FP32。若音质不满意可随时切回 FP32。"
            } else {
                "官方原版权重，音质基准；速度和功耗不如 INT8。"
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            when {
                ready && cloneReady -> "✅ 已就绪（含音色克隆编码器）"
                ready -> "✅ 已就绪（未下载克隆编码器）"
                else -> "⬇️ 未下载完整 · 已有 ${ModelStore.downloadedBytes(context, variant) / 1_048_576} MB"
            }
        )

        if (progress.running) {
            val frac = if (progress.total > 0) progress.done.toFloat() / progress.total else 0f
            LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth())
            Text(
                "${progress.file} · ${progress.done / 1_048_576}/${progress.total / 1_048_576} MB · ${progress.bytesPerSec / 1024} KB/s",
                style = MaterialTheme.typography.bodySmall,
            )
            OutlinedButton(onClick = { ModelDownloadService.cancel(context) }) { Text("取消下载") }
        } else {
            progress.error?.let { Text("下载失败：$it", color = MaterialTheme.colorScheme.error) }
            if (variant == ModelVariant.FP32) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    DownloadSource.entries.forEach { s ->
                        FilterChip(
                            selected = source == s,
                            onClick = { source = s; settings.downloadSource = s },
                            label = { Text(s.label, fontSize = 12.sp) },
                        )
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = withClone, onCheckedChange = { withClone = it })
                Spacer(Modifier.width(8.dp))
                Text("同时下载音色克隆编码器（约 45 MB）", style = MaterialTheme.typography.bodySmall)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!ready || (withClone && !cloneReady)) {
                    Button(onClick = { ModelDownloadService.start(context, variant, withClone) }) {
                        val mb = ModelStore.downloadMb(variant).takeIf { it > 100 } ?: if (variant == ModelVariant.FP32) 680L else 300L
                        Text(if (ModelStore.downloadedBytes(context, variant) > 0) "继续下载 / 更新" else "下载（约 $mb MB）")
                    }
                }
                if (ModelStore.downloadedBytes(context, variant) > 0) {
                    TextButton(onClick = {
                        EngineManager.releaseAsync()
                        ModelStore.deleteVariant(context, variant)
                        onChanged()
                    }) { Text("删除") }
                }
            }
        }
        Text(
            "也可以用 adb push 把官方两个 ONNX 目录放到：\n${ModelStore.root(context, variant).absolutePath}/",
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// --------------------------------------------------------------------------------- speak
@Composable
private fun SpeakCard(settings: AppSettings, refresh: Int) {
    val context = LocalContext.current
    val ui by SpeakService.state.collectAsStateWithLifecycle()
    var text by remember { mutableStateOf(settings.lastText) }
    var voices by remember { mutableStateOf(emptyList<VoicePrompt>()) }
    // re-read on refresh: a new clone selects itself
    var voiceId by remember(refresh) { mutableStateOf(settings.voiceId) }
    var speed by remember { mutableStateOf(settings.speed) }
    LaunchedEffect(refresh, settings.variant) {
        voices = withContext(Dispatchers.IO) { EngineManager.voices(context) }
    }
    val selected = voices.firstOrNull { it.id == voiceId } ?: voices.firstOrNull()
    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri: Uri? ->
        if (uri != null) SpeakService.saveWav(context, text, selected?.id, uri)
    }

    SectionCard("朗读（流式）") {
        VoicePicker(voices, selected) {
            voiceId = it.id
            settings.selectVoice(it)
        }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it; settings.lastText = it },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 120.dp),
            label = { Text("文本") },
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("语速 ${"%.2f".format(speed)}×", Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
            Slider(
                value = speed,
                onValueChange = { speed = (it * 20).toInt() / 20f },
                onValueChangeFinished = { settings.speed = speed },
                valueRange = 0.5f..2.0f,
                modifier = Modifier.weight(1f),
            )
        }
        if (selected != null) VolumeRow(settings, selected)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                enabled = text.isNotBlank() && selected != null,
                onClick = { SpeakService.speak(context, text, selected?.id) },
            ) { Text("朗读") }
            OutlinedButton(enabled = ui.busy, onClick = { SpeakService.stop(context) }) { Text("停止") }
            TextButton(
                enabled = text.isNotBlank() && selected != null && !ui.busy,
                onClick = { exportLauncher.launch("moss_tts_${System.currentTimeMillis()}.wav") },
            ) { Text("导出 WAV") }
        }
        if (ui.status.isNotEmpty()) Text(ui.status, style = MaterialTheme.typography.bodyMedium)
        ui.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        ui.stats?.let {
            Text(it.summary(), fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 16.sp)
        }
    }
}

/** Per-voice volume on top of the automatic loudness compensation; applies to the system TTS too. */
@Composable
private fun VolumeRow(settings: AppSettings, voice: VoicePrompt) {
    var offset by remember(voice.id) { mutableFloatStateOf(settings.volumeOffsetDb(voice.id)) }
    val auto = remember(voice.id) { Voices.autoGainDb(voice) }
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("音量 ${"%+d".format(offset.roundToInt())} dB", Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
            Slider(
                value = offset,
                onValueChange = { offset = it.roundToInt().toFloat() },
                onValueChangeFinished = { settings.setVolumeOffsetDb(voice.id, offset) },
                valueRange = AppSettings.VOLUME_MIN_DB..AppSettings.VOLUME_MAX_DB,
                steps = (AppSettings.VOLUME_MAX_DB - AppSettings.VOLUME_MIN_DB).toInt() - 1,
                modifier = Modifier.weight(1f),
            )
        }
        Text(
            buildString {
                if (auto >= 0.5f) append("这个音色的参考录音偏小声，已自动提高 ${"%.1f".format(auto)} dB；")
                append("滑块只调「${voice.displayName}」，系统 TTS 同样生效，带限幅不会爆音。")
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun VoicePicker(voices: List<VoicePrompt>, selected: VoicePrompt?, onSelect: (VoicePrompt) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("音色", Modifier.width(48.dp))
        OutlinedButton(onClick = { open = true }, enabled = voices.isNotEmpty()) {
            Text(selected?.let { "${it.displayName} · ${voiceSubtitle(it)}" } ?: "（模型未就绪）")
        }
    }
    if (open) {
        VoiceDialog(voices, selected?.id, onDismiss = { open = false }) {
            onSelect(it)
            open = false
        }
    }
}

private fun voiceSubtitle(v: VoicePrompt): String = when {
    !v.builtin -> "克隆 · ${"%.1f".format(v.frames * 0.08)} 秒"
    else -> VoiceLang.label(VoiceLang.of(v)) + when {
        v.group.contains("Female", ignoreCase = true) -> "女声"
        v.group.contains("Male", ignoreCase = true) -> "男声"
        else -> ""
    }
}

/**
 * Voice list as a dialog. (A DropdownMenu with 19+ entries is taller than the screen; with the
 * edge-to-edge layout Android 15+ enforces, its last entries ended up behind the navigation bar.)
 * Dialog windows stay inside the system bars, the list scrolls, and it opens at the current voice.
 */
@Composable
private fun VoiceDialog(voices: List<VoicePrompt>, selectedId: String?, onDismiss: () -> Unit, onPick: (VoicePrompt) -> Unit) {
    val rows: List<Any> = remember(voices) {
        val sections = listOf(
            "我的克隆音色" to voices.filter { !it.builtin },
            "中文" to voices.filter { it.builtin && VoiceLang.of(it) == VoiceLang.ZH },
            "英文" to voices.filter { it.builtin && VoiceLang.of(it) == VoiceLang.EN },
            "日文" to voices.filter { it.builtin && VoiceLang.of(it) == VoiceLang.JA },
        )
        sections.filter { it.second.isNotEmpty() }.flatMap { (title, list) -> listOf<Any>(title) + list }
    }
    val initial = remember(rows, selectedId) { rows.indexOfFirst { it is VoicePrompt && it.id == selectedId } }
    val listState = rememberLazyListState(initialFirstVisibleItemIndex = maxOf(0, initial - 2))
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } },
        title = { Text("选择音色") },
        text = {
            LazyColumn(state = listState, modifier = Modifier.heightIn(max = 520.dp)) {
                items(rows, key = { if (it is VoicePrompt) "v:" + it.id else "h:$it" }) { row ->
                    if (row is VoicePrompt) {
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .clickable { onPick(row) }
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            RadioButton(selected = row.id == selectedId, onClick = { onPick(row) })
                            Column(Modifier.weight(1f)) {
                                Text(row.displayName, style = MaterialTheme.typography.bodyLarge)
                                val auto = Voices.autoGainDb(row)
                                Text(
                                    voiceSubtitle(row) + if (auto >= 0.5f) " · 音量已补偿 +${"%.0f".format(auto)} dB" else "",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    } else {
                        Text(
                            row as String,
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
                        )
                    }
                }
            }
        },
    )
}

// --------------------------------------------------------------------------------- clone
@Composable
private fun CloneCard(settings: AppSettings, refresh: Int, onChanged: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var name by remember { mutableStateOf("我的音色") }
    var busy by remember { mutableStateOf(false) }
    var msg by remember { mutableStateOf<String?>(null) }
    val cloned = remember(refresh) { VoiceStore.list(context) }
    val canClone = remember(refresh, settings.variant) { ModelStore.canClone(context, settings.variant) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        busy = true
        msg = "处理中…"
        scope.launch {
            msg = withContext(Dispatchers.Default) {
                runCatching { VoiceCloner.clone(context, uri, name.ifBlank { "我的音色" }) }
                    .onSuccess { settings.selectVoice(it) }
                    .fold({ "已添加并选中音色「${it.displayName}」（${"%.1f".format(it.frames * 0.08)} 秒）" }, { "失败：${it.message}" })
            }
            busy = false
            onChanged()
        }
    }

    SectionCard("音色克隆") {
        Text(
            "选一段 5~10 秒、只有一个人说话、没有背景音乐的清晰录音。编码一次后保存，之后和内置音色一样快。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!canClone) Text("需要先下载音色克隆编码器（见「模型」）", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("音色名称") }, singleLine = true)
        Button(enabled = canClone && !busy, onClick = { picker.launch("audio/*") }) { Text("选择音频并克隆") }
        msg?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        cloned.forEach { v ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("${v.displayName}（${"%.1f".format(v.frames * 0.08)} 秒）", Modifier.weight(1f))
                TextButton(onClick = {
                    VoiceStore.delete(context, v.id)
                    onChanged()
                }) { Text("删除") }
            }
        }
    }
}

// --------------------------------------------------------------------------------- perf
@Composable
private fun PerfCard(settings: AppSettings) {
    val context = LocalContext.current
    var lm by remember { mutableIntStateOf(settings.lmThreads) }
    var prefill by remember { mutableIntStateOf(settings.prefillThreads) }
    var codec by remember { mutableIntStateOf(settings.codecThreads) }
    var spin by remember { mutableStateOf(settings.spinning) }
    var normalize by remember { mutableStateOf(settings.normalize) }
    var fixedSeed by remember { mutableStateOf(settings.fixedSeed) }
    var idle by remember { mutableIntStateOf(settings.idleUnloadMinutes) }
    var needRestart by remember { mutableStateOf(EngineManager.threadsNeedRestart(settings)) }

    SectionCard("性能与设置") {
        LabeledSlider("逐帧 LM 线程 $lm（推荐 2：单行矩阵向量乘，线程多了同步开销反而更慢、更费电）", lm.toFloat(), 1f..6f, 4) {
            lm = it.toInt(); settings.lmThreads = lm; needRestart = EngineManager.threadsNeedRestart(settings)
        }
        LabeledSlider("Prefill 线程 $prefill（每段文本一次的大矩阵乘，影响首音延迟）", prefill.toFloat(), 1f..6f, 4, onFinished = { EngineManager.releaseAsync() }) {
            prefill = it.toInt(); settings.prefillThreads = prefill
        }
        LabeledSlider("Codec 线程 $codec（与 LM 并行）", codec.toFloat(), 1f..4f, 2, onFinished = { EngineManager.releaseAsync() }) {
            codec = it.toInt(); settings.codecThreads = codec
        }
        val perfCores = remember { CpuAffinity.performanceCores }
        Text(
            if (perfCores > 0) "推理线程只跑在 $perfCores 个大核上（已排除小核）。LM + Codec 线程数之和不建议超过 $perfCores。"
            else "未识别到大小核结构，线程不做绑核。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        SwitchRow("线程自旋等待（更低延迟，略增功耗）", spin) {
            spin = it; settings.spinning = it; needRestart = EngineManager.threadsNeedRestart(settings)
        }
        SwitchRow("文本规整（数字/日期/单位读法）", normalize) { normalize = it; settings.normalize = it }
        SwitchRow("固定随机种子（同一句每次读法一致）", fixedSeed) { fixedSeed = it; settings.fixedSeed = it }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("空闲释放内存", style = MaterialTheme.typography.bodySmall)
            listOf(0, 5, 10, 30).forEach { m ->
                FilterChip(selected = idle == m, onClick = { idle = m; settings.idleUnloadMinutes = m }, label = { Text(if (m == 0) "不释放" else "${m}分") })
            }
        }
        if (needRestart) {
            Text("线程池设置需重启应用生效", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            Button(onClick = { restartApp(context) }) { Text("立即重启") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { EngineManager.preload() }) { Text("预加载") }
            OutlinedButton(onClick = { EngineManager.releaseAsync() }) { Text("释放内存") }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onFinished: () -> Unit = {},
    onChange: (Float) -> Unit,
) {
    Column {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Slider(value = value, onValueChange = onChange, onValueChangeFinished = onFinished, valueRange = range, steps = steps)
    }
}

@Composable
private fun SwitchRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Suppress("DEPRECATION")
private fun isPreferredEngine(context: Context): Boolean =
    Settings.Secure.getString(context.contentResolver, Settings.Secure.TTS_DEFAULT_SYNTH) == context.packageName

/** starts the first intent the system can open; false if none */
private fun startFirst(context: Context, vararg intents: Intent): Boolean {
    for (i in intents) {
        if (runCatching { context.startActivity(i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }.isSuccess) return true
    }
    return false
}

@Composable
private fun SystemTtsCard() {
    val context = LocalContext.current
    var preferred by remember { mutableStateOf(isPreferredEngine(context)) }
    // back from the system settings: show what was picked there
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { preferred = isPreferredEngine(context) }
    SectionCard("系统 TTS 引擎") {
        Text(
            "设为系统首选引擎后，阅读器（如 Anx Reader）、导航、语音助手和其他 App 都能用它朗读，同样是边生成边播放。也可以在任意 App 里选中文字 →「MOSS 朗读」。",
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            if (preferred) "✅ 已是系统首选引擎"
            else "还不是系统首选引擎。ColorOS 上这个设置藏得比较深，在系统设置里搜“文字转语音”最快；有 root 时也可以在语音助手里一键设置，或执行：\nsettings put secure tts_default_synth ${context.packageName}",
            style = MaterialTheme.typography.bodySmall,
            color = if (preferred) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedButton(onClick = {
            val ok = startFirst(
                context,
                Intent("com.android.settings.TTS_SETTINGS"),
                Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS),
                Intent(Settings.ACTION_SETTINGS),
            )
            if (!ok) Toast.makeText(context, "请在系统设置里搜索“文字转语音”", Toast.LENGTH_LONG).show()
        }) { Text("打开系统 TTS 设置") }
    }
}

/**
 * As the system engine, MOSS is started by other apps, often in the background (a reader with
 * the screen off, the assistant's voice call). ColorOS freezes background apps and blocks one app
 * from starting another unless allowed.
 */
@Composable
private fun BackgroundCard() {
    val context = LocalContext.current
    val pm = remember { context.getSystemService(PowerManager::class.java) }
    var unrestricted by remember { mutableStateOf(pm.isIgnoringBatteryOptimizations(context.packageName)) }
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { unrestricted = pm.isIgnoringBatteryOptimizations(context.packageName) }
    val details = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
    SectionCard("后台运行") {
        Text(
            if (unrestricted) "✅ 不受电池优化限制" else "受电池优化限制：熄屏或在后台朗读时，系统可能推迟或掐断它",
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            "给阅读器、语音助手当朗读引擎时，MOSS 是被别的 App 叫起来的。ColorOS 等系统除了电池优化还有自己的后台管理：在应用详情里打开“允许自动启动”“允许关联启动”，耗电管理选“允许完全后台行为”（名字随版本略有不同）。不设的话，别的 App 可能连不上朗读引擎（只显示文字不出声），熄屏朗读也可能被掐断。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!unrestricted) {
                Button(onClick = {
                    startFirst(
                        context,
                        Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${context.packageName}")),
                        Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS),
                        details,
                    )
                }) { Text("取消电池优化") }
            }
            OutlinedButton(onClick = { startFirst(context, details) }) { Text("应用详情") }
        }
    }
}

private fun restartApp(context: Context) {
    val intent = context.packageManager.getLaunchIntentForPackage(context.packageName)!!
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
    context.startActivity(intent)
    Process.killProcess(Process.myPid())
}
