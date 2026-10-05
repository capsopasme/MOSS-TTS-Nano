# MOSS-TTS-Nano for Android

在手机上**完全离线、边生成边播放**地运行 [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（0.1B 参数，48 kHz 立体声，多语种，音色克隆）。

基于官方 ONNX 导出（`MOSS-TTS-Nano-100M-ONNX` + `MOSS-Audio-Tokenizer-Nano-ONNX`）和 ONNX Runtime，针对骁龙 8 Gen 3（一加 Ace 5）调优，arm64 only。

## 功能

- **流式朗读**：首帧（80 ms 音频）一解码出来就开始播放，不等整句生成完
- **系统 TTS 引擎**：设为首选引擎后，阅读器 / 导航 / 其他 App 都能用，同样是流式输出，支持系统语速
- **任意 App 选中文字 →「MOSS 朗读」**，或分享文字到本 App
- **熄屏继续朗读**（前台服务 + wake lock）
- **音色克隆**：选一段 5–10 秒录音，编码一次存下来，之后和内置音色一样快
- 官方内置音色、导出 WAV、语速调节（保持音高）
- 模型应用内下载（Hugging Face / hf-mirror 国内镜像，断点续传），也可 `adb push`
- 可选 **INT8 量化模型包**（见下文）

## 为什么快：推理链路做了什么

官方仓库自带的 Android 示例是「整句生成 → 整句 codec 解码 → 写 WAV」。流式解码只存在于 Python 运行时里。本项目把流式链路完整移植到了 Android，并额外做了这些优化：

| 优化 | 说明 |
|---|---|
| 流式 codec 解码 | 使用有状态的 `moss_audio_tokenizer_decode_step.onnx`，codec 的 KV cache 在调用之间直接传递（与官方 `CodecStreamingDecodeSession` 一致） |
| LM 与 codec 并行 | LM（prefill → local frame → decode step）在合成线程上跑，codec + 音频输出在独立线程上跑，两者通过有界队列流水线并行 |
| 自适应 codec 批量 | 刚开始每帧立即解码（最低首音延迟），音频缓冲领先后自动合并为 2/4/8 帧一批，减少调用次数（官方策略） |
| 单一共享线程池 | prefill / decode / local 三个 LM 图共用一个 ORT 全局线程池；避免三个会话各自的线程池互相自旋抢核。codec 用独立的小线程池且不自旋 |
| 零拷贝 hidden state | decode_step 的 `global_hidden` 作为 pinned output 直接写进 local 图读取的同一块 direct buffer |
| 零分配热循环 | 每步输入（input row、past length、随机数、repetition mask）都是复用的 direct buffer 张量，原地修改；repetition mask 增量更新；KV cache 输出直接作为下一步输入，不经过 JVM |
| 只加载用到的图 | 不加载 decode_full / local_decoder / local_cached_step；克隆编码器只在克隆时临时加载 |
| AudioTrack 低起播阈值 | float PCM 直写，起播阈值设为 1 帧（80 ms），而不是默认的整个缓冲区 |
| 背压 | LM 最多领先播放约 6 秒，长文本不会一路狂算吃满 CPU 和内存 |
| ORT 取消 | 停止时通过 `RunOptions.setTerminate` 中断正在跑的图，响应即时 |

App 里每次朗读后会显示：首音延迟、RTF、prefill 耗时、LM 每帧耗时（decode/local 分项）、codec 耗时，方便调线程数。

## 安装

1. 从 [Actions](../../actions) 或 [Releases](../../releases) 下载 APK 安装。
2. 打开 App →「模型」→ 下载（FP32 约 760 MB，国内建议选 hf-mirror）。
   或者把官方两个目录 push 到 App 显示的路径：
   ```bash
   adb push MOSS-TTS-Nano-100M-ONNX     /sdcard/Android/data/io.github.capsopasme.mossnano/files/models/fp32/
   adb push MOSS-Audio-Tokenizer-Nano-ONNX /sdcard/Android/data/io.github.capsopasme.mossnano/files/models/fp32/
   ```
3. 输入文字 →「朗读」。首次朗读会加载模型并预热（几秒）。
4. 想全局使用：系统设置 → 文字转语音 → 首选引擎选 MOSS-TTS-Nano。

## INT8 量化模型包（可选）

`Actions → Build INT8 model pack → Run workflow` 会：

1. 从 Hugging Face 下载官方 ONNX；
2. 对三个 LM 图做动态 INT8 量化（MatMul/Gemm 权重 int8 per-channel，只量化常量权重矩阵，注意力和 embedding 保持 fp32），codec 不动；
3. 用 `tools/verify_onnx.py` 分别跑 FP32 / INT8，产出耗时 JSON 和两段 WAV（在 workflow artifact 里，可以先听再决定用不用）；
4. 发布到 `models-int8` release。

之后在 App 里切到 INT8 下载即可。INT8 走 ORT 的 ARM 点积 / i8mm 整数核，LM 权重内存约减半。**音质需要你亲耳确认**，不满意随时切回 FP32。

## 设置建议

- **LM 线程**：默认 4（X4 超大核 + 3 个 A720）。全局线程池在进程内固定，改了要重启 App（界面会提示）。
- **Codec 线程**：默认 2，和 LM 并行。
- **自旋等待**：开启时帧间延迟更低；在意功耗可以关。
- **空闲释放内存**：FP32 引擎常驻约 1 GB 以上原生内存，默认空闲 10 分钟自动释放。

## 项目结构

```
engine/   推理引擎（Android library，核心不依赖 Android API，可在桌面 JVM 上测试）
  MossTtsEngine.kt       流式推理主循环 + codec 线程
  CodecStreamDecoder.kt  有状态 codec 流式解码
  OrtSupport.kt          全局线程池 / 会话配置
  TextNormalizer.kt      官方 robust 规整的移植 + 中英文数字/日期/单位读法
  TextChunker.kt         官方分句逻辑的移植
  TimeStretch.kt         WSOLA 变速（系统 TTS 语速）
  cpp/                   SentencePiece JNI（与官方 tokenizer.model 完全一致的分词）
app/      UI、前台朗读服务、系统 TTS 服务、模型下载、音色克隆
tools/
  quantize_int8.py       INT8 量化
  verify_onnx.py         Python 版流式推理，用于对比 FP32/INT8
  mock/                  用与官方导出 I/O 完全一致的 mock 图在桌面 JVM 上测试引擎
```

## 测试

```bash
bash tools/mock/run_mock_test.sh
```

构建 ORT Java JNI（linux-x64）和 SentencePiece JNI，生成一组接口与官方导出一致、但带自检逻辑的 mock 图，然后跑真实的引擎代码，检查：KV cache 交接与 `past_valid_lengths` 一致性、pinned 输出、repetition mask 每段重置、codec 状态跨批次延续且每段重置、段间停顿、取消、中止、并发调用。CI 每次提交都会跑。

## 致谢 / 许可

模型与参考实现来自 [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（Apache-2.0）。分词使用 [google/sentencepiece](https://github.com/google/sentencepiece)，推理使用 [ONNX Runtime](https://github.com/microsoft/onnxruntime)。
