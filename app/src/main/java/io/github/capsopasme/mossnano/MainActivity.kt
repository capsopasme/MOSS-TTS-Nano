package io.github.capsopasme.mossnano

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.Process
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.capsopasme.mossnano.engine.VoicePrompt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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
                        EngineManager.release()
                        onChanged()
                    },
                    label = { Text(v.name) },
                )
            }
        }
        Text(variant.label, style = MaterialTheme.typography.bodySmall)
        if (variant == ModelVariant.INT8) {
            Text(
                "INT8 版由本仓库 GitHub Actions 从官方 ONNX 动态量化后发布：矩阵权重 int8（ARM 点积指令加速），LM 内存约减半。若音质不满意请切回 FP32。",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
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
                        Text(if (ModelStore.downloadedBytes(context, variant) > 0) "继续下载" else "下载（约 ${if (variant == ModelVariant.FP32) "760" else "550"} MB）")
                    }
                }
                if (ModelStore.downloadedBytes(context, variant) > 0) {
                    TextButton(onClick = {
                        EngineManager.release()
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
    var voiceId by remember { mutableStateOf(settings.voiceId) }
    var speed by remember { mutableStateOf(settings.speed) }
    LaunchedEffect(refresh, settings.variant) {
        voices = withContext(Dispatchers.IO) { EngineManager.voices(context) }
    }
    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri: Uri? ->
        if (uri != null) SpeakService.saveWav(context, text, voiceId, uri)
    }

    SectionCard("朗读（流式）") {
        VoicePicker(voices, voiceId) {
            voiceId = it
            settings.voiceId = it
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
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                enabled = text.isNotBlank() && voices.isNotEmpty(),
                onClick = { SpeakService.speak(context, text, voiceId) },
            ) { Text("朗读") }
            OutlinedButton(enabled = ui.busy, onClick = { SpeakService.stop(context) }) { Text("停止") }
            TextButton(
                enabled = text.isNotBlank() && voices.isNotEmpty() && !ui.busy,
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

@Composable
private fun VoicePicker(voices: List<VoicePrompt>, selectedId: String?, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    val selected = voices.firstOrNull { it.id == selectedId } ?: voices.firstOrNull()
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("音色", Modifier.width(48.dp))
        OutlinedButton(onClick = { open = true }, enabled = voices.isNotEmpty()) {
            Text(selected?.let { "${it.displayName} · ${it.group}" } ?: "（模型未就绪）")
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            voices.forEach { v ->
                DropdownMenuItem(
                    text = { Text("${v.displayName}  ·  ${v.group}") },
                    onClick = {
                        onSelect(v.id)
                        open = false
                    },
                )
            }
        }
    }
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
                    .fold({ "已添加音色「${it.displayName}」（${it.frames} 帧）" }, { "失败：${it.message}" })
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
                Text("${v.displayName}（${v.frames} 帧）", Modifier.weight(1f))
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
    var codec by remember { mutableIntStateOf(settings.codecThreads) }
    var spin by remember { mutableStateOf(settings.spinning) }
    var normalize by remember { mutableStateOf(settings.normalize) }
    var fixedSeed by remember { mutableStateOf(settings.fixedSeed) }
    var idle by remember { mutableIntStateOf(settings.idleUnloadMinutes) }
    var needRestart by remember { mutableStateOf(EngineManager.threadsNeedRestart(settings)) }

    SectionCard("性能与设置") {
        LabeledSlider("LM 线程 $lm（骁龙 8 Gen3 推荐 4 = 1 超大核 + 3 大核）", lm.toFloat(), 1f..8f, 6) {
            lm = it.toInt(); settings.lmThreads = lm; needRestart = EngineManager.threadsNeedRestart(settings)
        }
        LabeledSlider("Codec 线程 $codec（与 LM 并行）", codec.toFloat(), 1f..4f, 2) {
            codec = it.toInt(); settings.codecThreads = codec; EngineManager.release()
        }
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
            OutlinedButton(onClick = { EngineManager.release() }) { Text("释放内存") }
        }
    }
}

@Composable
private fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int, onChange: (Float) -> Unit) {
    Column {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Slider(value = value, onValueChange = onChange, valueRange = range, steps = steps)
    }
}

@Composable
private fun SwitchRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun SystemTtsCard() {
    val context = LocalContext.current
    SectionCard("系统 TTS 引擎") {
        Text(
            "设为系统首选引擎后，阅读器（如 Anx Reader）、导航和其他 App 都能用它朗读，同样是边生成边播放。也可以在任意 App 里选中文字 →「MOSS 朗读」。",
            style = MaterialTheme.typography.bodySmall,
        )
        OutlinedButton(onClick = {
            runCatching {
                context.startActivity(Intent("com.android.settings.TTS_SETTINGS").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }.onFailure { Toast.makeText(context, "请在 设置 → 系统 → 语言 → 文字转语音 中选择", Toast.LENGTH_LONG).show() }
        }) { Text("打开系统 TTS 设置") }
    }
}

private fun restartApp(context: Context) {
    val intent = context.packageManager.getLaunchIntentForPackage(context.packageName)!!
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
    context.startActivity(intent)
    Process.killProcess(Process.myPid())
}
