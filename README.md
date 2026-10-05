# MOSS-TTS-Nano for Android

在手机上**完全离线、边生成边播放**地运行 [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（0.1B 参数，48 kHz 立体声，多语种，音色克隆）。

基于官方 ONNX 导出（`MOSS-TTS-Nano-100M-ONNX` + `MOSS-Audio-Tokenizer-Nano-ONNX`）和 ONNX Runtime，针对骁龙 8 Gen 3（一加 Ace 5）调优，arm64 only。

## 功能

- **流式朗读**：首帧（80 ms 音频）一解码出来就开始播放，不等整句生成完
- **系统 TTS 引擎**：设为首选引擎后，阅读器 / 导航 / 其他 App 都能用，同样是流式输出，支持系统语速
- **任意 App 选中文字 →「MOSS 朗读」**，或分享文字到本 App
- **熄屏继续朗读**（前台服务 + wake lock），来电 / 其他 App 抢占音频焦点时自动停止
- **音色克隆**：选一段 5–10 秒录音，编码一次存下来，之后和内置音色一样快
- 18 个官方内置音色（中 6 / 英 5 / 日 7）、导出 WAV、语速调节（保持音高）
- 模型应用内下载（Hugging Face / hf-mirror 国内镜像，断点续传，固定到经过验证的版本），也可 `adb push`
- 默认使用 **INT8 模型包**（见下文），可随时切回官方 FP32

## 速度（实测）

同一句中文（7 秒音频），在 GitHub 的 ARM64 机器（Neoverse N2，Armv9，带 dot-product / i8mm，和骁龙 8 Gen 3 同一代指令集）上跑本仓库的 Kotlin 引擎：

| 模型 / 线程 | 首音延迟 | LM 每帧 | codec 每帧 | RTF |
|---|---|---|---|---|
| FP32，LM 4 线程（v0.1 默认）/ codec 1 线程 | 249 ms | 28.5 ms | 21.3 ms | 0.43 |
| FP32，v0.2 默认（LM 2 / prefill 4 / codec 2） | 256 ms | 26.3 ms | 11.6 ms | 0.40 |
| **INT8 v2，v0.2 默认** | **107 ms** | **12.7 ms** | **11.7 ms** | **0.21** |

（每帧 = 80 ms 音频，实时要求 < 80 ms。）INT8 v2 + 2 个 LM 线程比 FP32 + 4 线程每帧快 2.2 倍，同时少占两个核。只看 LM 图：旧 INT8 包 2 线程 18.0 ms/帧，INT8 v2 12.9 ms/帧。CI 每次构建模型包都会重新跑这组测试（`Build INT8 model pack` → `bench-arm`）。

## 为什么快：推理链路做了什么

官方仓库自带的 Android 示例是「整句生成 → 整句 codec 解码 → 写 WAV」。流式解码只存在于 Python 运行时里。本项目把流式链路完整移植到了 Android，并额外做了这些优化：

| 优化 | 说明 |
|---|---|
| 流式 codec 解码 | 使用有状态的 `moss_audio_tokenizer_decode_step.onnx`，codec 的 KV cache 在调用之间直接传递（与官方 `CodecStreamingDecodeSession` 一致） |
| LM 与 codec 并行 | LM（prefill → local frame → decode step）在合成线程上跑，codec + 音频输出在独立线程上跑，两者通过有界队列流水线并行 |
| 自适应 codec 批量 | 刚开始每帧立即解码（最低首音延迟），音频缓冲领先后合并为 2/4/8 帧一批（官方策略），领先 3 秒以上再合并到 16 帧：codec 每次调用有很大的固定开销（12 层滑窗注意力缓存，最长 1600 步），批量越大每帧越省 |
| 线程池按负载拆分 | 逐帧的 decode / local 图是单行矩阵向量乘，共用一个 2 线程的自旋全局池（实测 2 线程最快，4 线程反而慢 30%）；prefill 是每段一次的大矩阵乘，用独立的 4 线程不自旋池；codec 用独立的小池且不自旋 |
| 只用大核 | 推理线程（ORT 全局池、prefill 池、codec 池、调用线程）绑定到大核，排除最慢的小核簇：一个 worker 落到 A520 上，所有并行段都要等它。CPU 拓扑运行时从 sysfs 读取，同构 CPU 上不绑核 |
| 音频优先级 | ORT 的线程在加载时创建，继承加载线程的优先级，所以加载时临时提到音频优先级 |
| INT8 v2 模型包 | LM 全部矩阵权重 int8（含 local 图里展开的 17 步），local 图的 text head 只算用得到的 2 个 logit，见下文 |
| 零拷贝 hidden state | decode_step 的 `global_hidden` 作为 pinned output 直接写进 local 图读取的同一块 direct buffer |
| 零分配热循环 | 每步输入（input row、past length、随机数、repetition mask）都是复用的 direct buffer 张量，原地修改；repetition mask 增量更新；KV cache 输出直接作为下一步输入，不经过 JVM |
| 只加载用到的图 | 不加载 decode_full / local_decoder / local_cached_step；克隆编码器只在克隆时临时加载 |
| AudioTrack 低起播阈值 | float PCM 直写，起播阈值设为 1 帧（80 ms），而不是默认的整个缓冲区 |
| 背压 | 队列 + AudioTrack 阻塞写限制 LM 最多领先播放十几秒，长文本不会无限占内存 |
| ORT 取消 | 停止时通过 `RunOptions.setTerminate` 中断正在跑的图，响应即时；释放模型时会先取消正在进行的合成，不会卡住界面 |

App 里每次朗读后会显示：首音延迟、RTF、prefill 耗时、LM 每帧耗时（decode/local 分项）、codec 耗时，方便调线程数。

## 安装

1. 从 [Releases](../../releases) 下载 APK 安装（用固定的 release key 签名，之后的版本可以直接覆盖升级）。
2. 打开 App →「模型」→ 下载（INT8 约 540 MB / FP32 约 720 MB，另有可选的克隆编码器 45 MB；国内建议选 hf-mirror）。
   FP32 也可以把官方两个目录 push 到 App 显示的路径：
   ```bash
   adb push MOSS-TTS-Nano-100M-ONNX     /sdcard/Android/data/io.github.capsopasme.mossnano/files/models/fp32/
   adb push MOSS-Audio-Tokenizer-Nano-ONNX /sdcard/Android/data/io.github.capsopasme.mossnano/files/models/fp32/
   ```
3. 输入文字 →「朗读」。首次朗读会加载模型并预热（几秒）。
4. 想全局使用：系统设置 → 文字转语音 → 首选引擎选 MOSS-TTS-Nano。

## INT8 模型包

`Build INT8 model pack` workflow（`tools/quantize_int8.py`）从 Hugging Face 下载官方 ONNX（固定 revision），然后：

1. **折叠 `Identity(权重)`**：local 图每帧把 local transformer 展开 17 次，torch 导出时把每次复用的权重都写成第一份权重的 `Identity`。`quantize_dynamic` 只量化 B 是常量的 MatMul，所以旧版 INT8 包里 17 步中有 16 步仍是 FP32（约占每帧权重读取量的 85%）。v2 先折叠这些 Identity，再量化，且量化后若还有带常量权重的 FP32 MatMul 就直接报错。
2. **裁剪 text head**：local 图为了在「继续 / 结束」两个 token 之间做选择，每帧要算完整的 16384 维 text logits。v2 把权重裁成只含这 2 列（数学上完全等价）。
3. 对三个 LM 图做动态 INT8 量化（MatMul 权重 int8 per-channel，走 ORT 的 ARM 点积 / i8mm 整数核）；注意力和 embedding 保持 fp32，codec 不动。

每次构建都会自动检查（结果写在 workflow 的 annotations 里）：

- `tools/check_int8_pack.py`：图改写无损（与官方 FP32 local 图在相同输入下采样出完全相同的帧）；INT8 与 FP32 的 hidden state 余弦相似度（teacher forcing，均值 0.998）。
- `tools/quality_int8.py`：让每个模型包自由生成，再用官方 FP32 模型给生成的每个音频 token 打分（FP32 似然，用官方 `local_cached_step` 图）。当前结果：FP32 自身 3.504 nat/token，INT8 v2 3.559（+1.6%），旧 INT8 包 3.537。
- `tools/bench_onnx.py` 和 `tools/mock/run_real_model.sh`：在 ARM64 机器上测 FP32 / 旧 INT8 / 新 INT8 的速度（上面的表格）。
- workflow artifact 里有 FP32 和 INT8 各一段 WAV，可以下载对比试听。

发布到 `models-int8-v2` release，App 按文件大小校验，旧版本的文件会自动重新下载。

## 设置建议

- **逐帧 LM 线程**：默认 2。逐帧计算是单行矩阵向量乘，超过 2 个线程后算子间同步的开销比多出来的核还大，而且每个自旋线程都占满一个核。全局线程池在进程内固定，改了要重启 App（界面会提示）。
- **Prefill 线程**：默认 4，影响首音延迟；不自旋，段与段之间不占 CPU。
- **Codec 线程**：默认 2，和 LM 并行。
- **自旋等待**：开启时帧间延迟更低；在意功耗可以关。
- **空闲释放内存**：引擎常驻数百 MB 原生内存，默认空闲 10 分钟自动释放。

## 项目结构

```
engine/   推理引擎（Android library，核心不依赖 Android API，可在桌面 JVM 上测试）
  MossTtsEngine.kt       流式推理主循环 + codec 线程
  CodecStreamDecoder.kt  有状态 codec 流式解码
  OrtSupport.kt          全局线程池 / 会话配置
  CpuAffinity.kt         大核识别 + 绑核
  TextNormalizer.kt      官方 robust 规整的移植 + 中英文数字/日期/单位读法
  TextChunker.kt         官方分句逻辑的移植
  TimeStretch.kt         WSOLA 变速（系统 TTS 语速）
  cpp/                   SentencePiece JNI（与官方 tokenizer.model 完全一致的分词）+ 绑核
app/      UI、前台朗读服务、系统 TTS 服务、模型下载、音色克隆
tools/
  quantize_int8.py       INT8 模型包（Identity 折叠 + text head 裁剪 + 量化）
  check_int8_pack.py     图改写无损性 + 量化误差检查
  quality_int8.py        FP32 似然打分
  bench_onnx.py          各个图的速度测试
  verify_onnx.py         Python 版流式推理，产出可试听的 WAV
  mock/                  在桌面 JVM（x86_64 / aarch64）上跑真实的引擎代码
```

## 测试

```bash
bash tools/mock/run_mock_test.sh                       # mock 图，CI 每次提交都跑
bash tools/mock/run_real_model.sh <模型目录> out.wav    # 真实模型（FP32 或 INT8 包）端到端
```

`run_mock_test.sh` 构建 ORT Java JNI 和 SentencePiece JNI，生成一组接口与官方导出一致、但带自检逻辑的 mock 图，然后跑真实的引擎代码，检查：KV cache 交接与 `past_valid_lengths` 一致性、pinned 输出、repetition mask 每段重置、codec 状态跨批次延续且每段重置、段间停顿、取消、中止、并发调用、音频输出失败不卡死、合成中释放引擎、绑核、文本规整用例。

## 发布

推送 `v*` tag 会构建并发布 APK。签名用仓库 secrets（`SIGN_KEY_BASE64` / `SIGN_KEY_PWD` / `SIGN_KEY_ALIAS`，也认 `SIGNING_KEYSTORE_BASE64`、`RELEASE_KEYSTORE_BASE64` 等常见命名）；tag 构建找不到 release key 时会直接失败，不会发布 debug 签名的 APK。release 附带 `SHA256SUMS.txt`，签名证书指纹写在 workflow 的 annotations 里。

## 致谢 / 许可

模型与参考实现来自 [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)（Apache-2.0）。分词使用 [google/sentencepiece](https://github.com/google/sentencepiece)，推理使用 [ONNX Runtime](https://github.com/microsoft/onnxruntime)。
