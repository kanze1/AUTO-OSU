# AUTO-OSU

[![release](https://img.shields.io/github/v/release/kanze1/AUTO-OSU?label=%E4%B8%8B%E8%BD%BD&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/releases)
[![stars](https://img.shields.io/github/stars/kanze1/AUTO-OSU?style=flat&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/stargazers)
[![tests](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml/badge.svg)](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT%20%2B%20%E7%BD%B2%E5%90%8D-4fb8ff)](LICENSE)

**丢进一首歌，一分钟后拿到一张能直接打的 osu!standard 谱面。**

[English](README.en.md) · [下载](https://github.com/kanze1/AUTO-OSU/releases) · [怎么用](#怎么用) · [效果与局限](#效果与局限) · [开发排期](#开发排期) · [参与协作](#分支管理) · [交流群](#交流与反馈)

**QQ 交流群：1124526648**（搜索群号加入）—— 使用反馈、谱面交流、模型实验与开源协作。

![AUTO-OSU 主界面](docs/screenshot_zh.png)

<details>
<summary>亮色主题 / 批量结果</summary>

![亮色主题](docs/screenshot_en.png)

![批量结果](docs/screenshot_busy.png)

</details>

## 为什么做这个

我打 osu! 很菜，但是很爱玩。最难受的事情是：想玩的歌没有人做图，自己又不会做图。
于是就有了 AUTO-OSU：把喜欢的歌丢进去，一分钟后就能开打。

已发布应用为 **0.2.0**，当前源码候选版为 **0.3.0rc2**，节奏和坐标模型仍为 **v0**。已经能出让我自己愿意打完整首的图：节奏卡在鼓点上，跳和串都是从十几万张 ranked / approved / loved 谱面里学来的，
四个难度一次出齐。接下来它会继续变强：谱师的设计意图、刻意的 highlight、更长的滑条、变速歌的多红线，都在路线图上。

如果你也是"就想用喜欢的歌打一把"的人，欢迎拿去用、提 issue、一起改。
觉得有用的话点个 ⭐ **Star**，对我很重要。—— kanzei

本项目面向本地游玩、练习和生成实验。分享谱面时请明确标注使用了 AUTO-OSU，保留生成来源信息，不要冒充人工制谱。
osu! 当前的 [Ranked AI policy](https://osu.ppy.sh/wiki/en/Ranking_criteria#ai-policy) 要求物件、音效和 timing 由人工直接制作，
因此请勿将本工具生成的谱面用于申请 Ranked；人工检查或注明 AI 来源也不改变这一要求。

## 怎么用

### 免安装（Windows）

1. 到 [Releases](https://github.com/kanze1/AUTO-OSU/releases) 下载 `AUTO-OSU-<版本>-win64-cpu.zip`（模型已内置），解压到任意位置。
2. 双击 `AUTO-OSU.exe`。
3. 把歌拖进窗口，勾选难度，点 **生成谱面**。
4. `.osz` 写到 exe 旁边的 `output` 文件夹，并默认直接在 osu! 里打开（自动导入）。打开 osu! 就能在歌曲列表里找到。

不需要显卡：3 分钟的歌、一个难度，在现代 CPU 上约 1 分钟（16 核实测 70 秒；RTX 4090 上 16 秒）。
Windows 10 / 11，64 位。

### 一键配置 GPU 加速

在窗口的「计算设备」下点击 **自动配置 GPU 加速**。程序会自动准备 uv、独立的 Python 3.12 和适合显卡驱动的 CUDA 版 PyTorch，完成真实 CUDA 运算检查后立即启用，无需重启窗口。

- 无需预装 Python、uv 或 CUDA Toolkit；需要 NVIDIA 显卡及驱动。
- 首次联网下载约数 GB；安装进度和详细日志直接显示在窗口中，可以取消。
- 环境保存在 `%LOCALAPPDATA%\AUTO-OSU\runtime`，不修改系统 Python。重新配置失败时保留原来可用的环境。
- `auto` 优先使用可用的 GPU；`cpu` 始终使用 CPU；显式选择 `cuda` 时，不可用会报出原因。

### 整个文件夹批量生成

切换 **文件夹批量**，选择或拖入目录，按需勾选「包含子文件夹」，再点击 **批量生成**。程序会扫描支持的音频和视频，并在队列中显示每首歌的状态。

每首歌保存到独立子目录，同名歌曲不会覆盖；坏文件记录错误后继续处理下一首。批次结束会生成 `batch-report-*.json`，包含成功输出、失败原因和运行设备。点击「停止后续任务」会完成当前歌曲，再停止剩余队列。批量模式由你选取生成的 `.osz` 导入 osu!。

### 支持什么音频

输入会先标准化再进流水线，所以格式基本不挑：

- 音频：mp3、ogg、wav、flac、m4a / aac、wma、opus、aiff、ape、alac 等；
- 视频：mp4、mkv、webm、mov、avi 等，自动抽出音轨；
- 打进 `.osz` 的音频保证 osu! 能播：mp3 和 ogg-vorbis 原样保留；其他格式转 mp3，码率跟着源走（无损源 320 kbps，有损源按原码率向上取整，192 kbps 起）。
- 源文件里的封面图会直接当谱面背景；视频源没有封面就截一帧。歌名、歌手也从标签里读。

程序自带 ffmpeg，不用另外安装。

### 界面

| 区域 | 说明 |
| --- | --- |
| 歌曲 | 单曲转换 / 文件夹批量，支持拖放；下方显示处理队列。 |
| 难度 | Easy / Normal / Hard / Insane 可多选，全部打进同一个 `.osz`。默认 Hard + Insane。 |
| 输出 | 保存目录；「生成后自动导入 osu!」会直接打开 `.osz`，等于双击它。 |
| 模型 | 显示模型是否就绪；缺失时一键下载（自动校验）。 |
| 计算设备 | 显示实际可用的 CUDA、显卡与显存；支持重新检测和一键配置 GPU 加速。 |
| 顶部封面 | 中 / 英切换，亮 / 暗切换。设置、上次的歌和目录都会记住。 |

**高级选项**（点「高级选项」展开）：

| 选项 | 说明 |
| --- | --- |
| 随机种子 | 同一首歌换个数字得到另一版摆放；同一个种子结果可复现。 |
| BPM / 偏移 | 留空自动检测；检测错了（常见于变速歌或前奏很空的歌）手动填。偏移单位 ms。 |
| 谱师名 | 写进 `.osu` 的 Creator，默认 AUTO-OSU。 |
| 星级条件 | 给模型的难度提示，留空按难度默认：Easy 2.0 / Normal 3.2 / Hard 4.5 / Insane 5.5。源码版生成后另行显示实际星级（NM / stable）；条件与实测值可能不同。 |
| 摆放质量 | 坐标模型的扩散步数：快速 50 / 标准 100 / 精细 200。标准档够用。 |
| 生成引擎 | 「AI 模型」是正常模式；「纯规则」不用模型、几秒出图，只在没模型或想对比时用。 |
| 试听 mp3 | 另存一个原曲压低音量、每个物件加点击声的 mp3，不开 osu! 也能听节奏对不对。 |

### 难度与 highlight 控制

展开高级选项，打开 **难度与 highlight 控制**：可填目标实测星级、密度、跨度倍率，以及原音频中的 highlight 起止秒数。目标星级最多尝试 3 个候选，排除结构错误和孤立难度尖峰后按实测值选择；达不到时会明确提示。新配置默认自动 highlight，也可选手动、关闭或保留原有 kiai；已经保存的模式继续保留。其他控制留空并选择原有 kiai，可恢复原版生成方式。

手动 highlight 使用同一套区间驱动密度、跨度、combo、音效和 kiai；滑条速度变化独立选择。自动模式结合响度、打击与和声 onset 提出小节对齐的区间，对比不足则不添加。维护者已检查前三首 A/B/C/D 对照，认为整体自然，第一首更偏好手动版。自动定位和高星目标仍保留实验标记。

```powershell
python -m autoosu "song.mp3" -d Insane --target-star 6.5 --candidates 3
python -m autoosu "song.mp3" -d Insane --highlight 30:55:1 --highlight 80:100:0.7
python -m autoosu "song.mp3" -d Insane --highlight-mode auto
python -m autoosu "song.mp3" -d Insane --control-plan "plan.json"
```

计划支持导入 / 导出，CLI、GUI、独立 worker 和文件夹批量共用；某首歌短于手动区间时该首报错，批量继续处理下一首。跨度控制和候选生成会增加耗时。详见[实现与验收说明](docs/generation-controls.md)及[0.3.0rc1 候选版与高难度对照](docs/release-0.3.0rc1.md)。

### 谱面偏好（实验）

在同一个控制窗口选择 **综合（默认） / 偏跳跃 / 偏连打、短串**。条件版仍用 v0 权重：跳跃优先保留原节奏并调整落点跨度，连打用参考谱面的局部密度引导更连续的点按。手工密度、曲线和跨度优先；自动 highlight 仍默认开启，也可自行切换。

偏好模式最多生成 3 个候选，在相近实际星级下检查倾向变化；没有满足时明确提示，不能保证每首都有效。这里的连打包含短串，不承诺长串。需要两个模型，会增加生成耗时。带标签的新模型尚未训练，见下方排期和[偏好验收记录](docs/skill-preferences.md)。

```powershell
python -m autoosu "song.mp3" -d Insane --skill-preference jumps
python -m autoosu "song.mp3" -d Insane --skill-preference streams --target-star 6.5
```

### 命令行参数

`python -m autoosu 歌曲 [选项]`，exe 也认同样的参数（`AUTO-OSU.exe 歌曲.mp3 -d Hard`）。

```powershell
python -m autoosu --setup-runtime
python -m autoosu --check-cuda
python -m autoosu "D:\Music" --recursive -d Hard Insane -o "D:\Beatmaps"
```

| 选项 | 说明 |
| --- | --- |
| `-d Easy Normal Hard Insane` | 要生成的难度 |
| `-o 目录` | 输出目录，默认 `out` |
| `--seed N` | 随机种子 |
| `--bpm` / `--offset` | 手动 BPM / 红线偏移（ms） |
| `--title` / `--artist` / `--creator` | 覆盖元数据（默认读音频标签，或「歌手 - 歌名」文件名） |
| `--star X` | 星级条件 |
| `--coord-steps N` | 扩散步数，默认 100 |
| `--cfg-scale X` | 坐标模型的 classifier-free guidance，默认 1.0 |
| `--temperature` / `--density` / `--density-bias` / `--decode-steps` | 节奏模型采样参数：温度、目标每小节物件数、"不放"偏置（负数更密）、解码轮数 |
| `--device auto\|cuda\|cpu` | 计算设备 |
| `--setup-runtime` | 使用 uv 自动安装并验证应用专属 GPU 环境 |
| `--check-cuda` | 检查当前实际推理环境（包括自动安装的环境） |
| `--check-source 路径` | 检查 `.osu` / `.osz` / 文件夹的新图水印、来源声明与本机记录（源码开发版） |
| `--source-report 报告.json` | 保存来源检查的 JSON 报告 |
| `--records-dir 目录` | 指定本机生成记录目录，生成与检查使用同一目录 |
| `--recursive` | 输入为目录时，包含子文件夹 |
| `--rules` | 纯规则模式 |
| `--rhythm-model` / `--coord-model` | 指定模型文件；不指定则在 `models/` 里找 |
| `--no-coord-model` | 只用节奏模型，摆放走规则 |
| `--download` | 缺模型时从 GitHub Release 下载 |
| `--osu-shift 26` | 写入 `.osu` 时物件提前多少 ms |
| `--preview` | 另存试听 mp3 |
| `--debug-plot` | 另存分析图：响度与 kiai 段、onset 与拍线、各难度选中的音符 |
| `--dump-events` | 打印每个物件的时间、类型、拍位 |

### 检查谱面来源（源码开发版）

源码开发版可点击窗口底部的「检查谱面来源」，选择谱面、谱包或文件夹，另存 JSON 报告。命令行：

```powershell
python -m autoosu --check-source "map.osz" --source-report "source-report.json"
python -m autoosu --check-source "D:\Beatmaps" --recursive
```

新图默认添加实验性内容水印，并保存模型身份、参数、内容指纹和本机生成记录。水印只微调圆圈坐标，每个坐标轴最多 1 像素；删标签、改标题、重新打包后仍可检查。少于 64 个不同拍点的圆圈时跳过水印并显示原因。检查结果区分「检出内容水印」「包含来源声明」「匹配本机记录」「无法判断」。

水印是公开的来源标记，可以被复制、移除或由 fork 绕过，不能认证模型确实执行过。未检出不代表人工制作。**旧图不做追溯检测**；[旧版实验](docs/detection-v0-screen.md) 仅保留记录。算法、误报与编辑测试见[新图水印说明](docs/generation-watermark.md)。

记录默认保存在 `~/.autoosu/provenance`，可用 `--records-dir` 或 `AUTOOSU_RECORDS` 修改。详细范围与验证见[来源检查说明](docs/source-check.md)。已发布的 0.2.0 下载不含此功能；开发版需要匹配版本的独立 GPU 环境。

### Python 安装

```bash
git clone https://github.com/kanze1/AUTO-OSU
cd AUTO-OSU
python -m venv .venv && .venv\Scripts\activate       # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -e .[gui]
python -m autoosu --setup-runtime                     # 可选：自动准备独立 GPU 环境
python -m autoosu --download                          # 第一次下载模型（约 320 MB）
python -m autoosu                                     # 打开图形界面
python -m autoosu "歌曲.mp3" -d Hard Insane -o out    # 命令行
```

Python 3.10 及以上；实际星级测量需要 Python 3.11 及以上。macOS / Linux 用这种方式运行，exe 只提供 Windows 版。

源码版使用固定的 `rosu-pp-py 4.0.2` 计算实际星级。生成后在日志显示实测值，在 `.osz` 内保存 `autoosu-evaluation.json`，输出目录另存同名 `.evaluation.json`；包含 aim / speed、分段 strain 和结构检查。批量报告也记录每个难度的实测值。计算器不可用时明确显示未测出，不影响保存谱面。[评估说明](docs/difficulty-evaluation.md)。

## 效果与局限

- **节奏模型。** v0 训练评估记录的生成 onset F1 为 0.963。评估使用原谱 timing 和真实局部密度，默认抽取 6 条、每条最多 512 tick；这个结果不等于任意歌曲自动生成的端到端质量，也不代表超过人工制谱。
- **坐标与滑条。** 坐标模型从纯噪声生成坐标，学习跳、串和滑条形状；滑条导出前经过场地与长度贴合。历史测试样本的路径出界率为 0%，仍需结合实际歌曲检查手感。
- **还差的地方。** 一条红线；滑条长度不是模型输入，快歌上的长滑条偶尔被缩短（会补绿线保证时长正确）；
  打击音效只有基于鼓的简单 whistle / clap / finish；没有 storyboard。游玩或分享前请在编辑器里检查 timing、可读性和手感。
- **来源识别。** 源码开发版为新图添加内容水印，区分双模型、两种混合与纯规则的来源声明，并检查本机记录。旧图不作模型归因；旧版 0.2.0 的 `ai-generated` 标签也用于纯规则结果，不能据此确认用过模型。

## 原理

![架构图](docs/architecture.png)

**分析与定时（规则）。** 打击 / 旋律分离，分频段（底鼓 / 军鼓 / 镲）算 onset 包络；tempogram 估 BPM 并纠正倍频；
在波形上做 1 ms 精度的偏移校准；用底鼓、和声变化和响度找小节起拍；按响度分段找 kiai。
物件统一比音频瞬态提前 26 ms 写入，这是 ranked 谱面的普遍惯例，玩家的偏移设置都按它校准。

**节奏模型** `rhythm_v0.pt`，约 2900 万参数。1/4 拍网格上的双向 Transformer：每个 tick 看 ±80 ms 的梅尔频谱、
在小节里的位置、局部响度，加上要求的星级 / CS / AR / OD / HP，预测六类之一（无 / 圈 / 滑条头 / 身 / 尾 / 转盘），
MaskGIT 式 12 轮并行解码。

**坐标模型** `coord_v0.pt`，约 1.3 亿参数。结构是 [osu-diffusion](https://github.com/OliBomby/osu-diffusion) 的 DiT-B，
本项目用完整的 1000 步噪声计划从零训练。输入是物件 token 序列（圈、滑条头、锚点、滑条尾、转盘，各带时间），
从纯噪声去噪出每个点的 x / y；条件只有星级和 CS，训练时一半样本把"到上一个点的距离"置零，所以它会自己决定跨度。

**滑条贴合。** 模型画形状，长度由节奏决定：每条滑条围绕滑条头缩放到要求长度；会出场地就先镜像、再旋转，
实在放不下才缩短并补一条本地绿线。快歌会按 BPM 压低 SliderMultiplier。

### 训练记录

![训练曲线](docs/training_curves.png)

| 模型 | 数据 | 硬件 | 步数 | 训练时长 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 节奏 masked v0 | 139,582 张 osu!standard 谱面（[project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps)） | 2 × RTX 5880 Ada | 60k，batch 128 | 4.5 h | 取第 40k 步：生成 onset F1 0.963，密度误差 0.10 |
| 坐标 DiT-B v0 | 同一语料的 140,018 张（ORS 布局） | 2 × RTX 5880 Ada | 200k，batch 128 | 8.5 h | 最终 loss 0.125；出界 0%，同种子 5 万 / 10 万 / 20 万步布局几乎一致 |

表中为历史实验记录；F1 的条件与采样范围见上方「效果与局限」，出界和布局比较来自已测样本。

自回归版本的节奏模型也训过，因果注意力听不到后面的音频、还爱用转盘逃课，被掩码版淘汰了。
完整实验记录（含失败的路线）在 [docs/rhythm_model_design.md](docs/rhythm_model_design.md)，
wandb 项目：[autoosu-rhythm](https://wandb.ai/kanzei/autoosu-rhythm)、[autoosu-coords](https://wandb.ai/kanzei/autoosu-coords)。

## 自己训练 / 打包

训练用到的全部代码都在仓库里：`autoosu/ml/prepare_data.py`（HF 分片 → 特征与标签）、`autoosu/ml/train.py`（节奏模型，`torchrun` 多卡）、
`coord/` + `scripts/coord_make_ors.py`（坐标模型，accelerate）、`scripts/server_*.sh`（服务器流程）、`scripts/coord_export.py`（导出发布文件）。
两个模型文件都能用 `torch.load(weights_only=True)` 加载，不含 pickle 代码。

打 exe：

```powershell
pip install -e .[build]
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # dist/AUTO-OSU-<版本>-win64-cpu.zip
```

用装了 CUDA 版 torch 的环境加 `-Venv .venv-gpu -Suffix cuda` 可打显卡版。`python scripts/make_icon.py 头像.png` 生成窗口头像和 exe 图标。

## 常见问题

**模型下载失败？** 手动从 [models-v0](https://github.com/kanze1/AUTO-OSU/releases/tag/models-v0) 下载 `rhythm_v0.pt` 和 `coord_v0.pt`，
放到 exe 旁边的 `models/` 文件夹（或 `~/.autoosu/models/`）。

**杀毒软件报毒？** PyInstaller 打的包常被误报。可以用 Python 方式运行，或自己按上面的步骤打包。

**BPM 或偏移不对？** 在高级选项里手动填。变速歌目前只会给一条红线。

**osu! 没有自动打开？** 说明 `.osz` 没有关联到 osu!，把生成的 `.osz` 拖进 osu! 窗口即可。

**生成很慢？** CPU 一个难度约一分钟属于正常；摆放质量选「快速」能快一倍；有 NVIDIA 显卡可在窗口点击「自动配置 GPU 加速」。

**某个格式解不开？** 先确认文件本身能播放；程序会先用 libsndfile 再用自带的 ffmpeg 解码，都失败会提示具体原因。

## 开发排期

以下按优先级和依赖顺序推进，不设时间承诺；每项以验收结果、PR 和 Release 为准。

| 优先级 / 顺序 | 任务 | 完成标准 | 状态 |
| --- | --- | --- | --- |
| P0 · 文档与协作 | 用途与来源说明、中英文 README、分支与协作约定、交流群 | 文档同步，计划功能与已发布功能分开说明 | 已更新 |
| P0 · 新图来源 | 默认内容水印、生成记录、模型身份与内容指纹；旧图不追溯 | 支持 `.osu` / `.osz`；验证去标签、编辑、误报与星级影响 | 水印和检查器已接入源码；真实客户端另存与试玩待验证 |
| P1 · 难度控制 | 实际星级、局部密度曲线、两次采样的跨度控制、最多 3 个候选筛选 | 对比目标与实际星级，先验证 6–7★，再扩展 7–8★，同时检查局部难度峰值和手感 | 已接入并完成初步人工检查；高难度对照继续验收 |
| P1 · 段落表现 | 手动 highlight 与自动区间建议，协调密度、跨度、音效和 kiai | 先验证手工区间，再对照人工标注检验自动定位；低对比可不添加 | 默认自动、模式可选；初步人工反馈自然，自动位置标签待补 |
| P1 · 谱面偏好 S1 | 综合 / 偏跳跃 / 偏连打，先接现有条件与候选筛选 | 同歌、相近实际星级验收；未满足明确提示 | 已接入源码，保持实验状态；见[实测记录](docs/skill-preferences.md) |
| P2 · 标签数据 M1.1 | 标签定义、来源与人工复核、按歌曲分组划分 | 缺标签与负标签分开，统计不同星级覆盖，冻结独立测试集 | 已排期，未开始 |
| P2 · 条件训练 M1.2 | 小规模标签训练，对比节奏 / 坐标 / 双模型条件 | 优于当前条件版且无标签模式不过退，再扩大训练 | 依赖 M1.1，未训练 |
| P2 · 独立验收 M1.3 | 同星级技能响应、高难度、highlight 和盲测 | 冻结模型后验收新歌曲，分开自动 / 参考 timing 与真实试玩 | 依赖 M1.2 |
| P2 · 新权重发布 M1.4 | 标签界面、模型兼容和下载校验、候选包 | 验收通过后再替换默认权重，保留 v0 回退 | 依赖 M1.3，尚未发布 |

检测范围仅限今后主动标记的新图，不再推进旧图统计归因。未检出显示“无法判断”，不代表人工制作。
当前状态见[任务队列](docs/TASKS.md)，重训范围和通过标准见[带标签模型排期](docs/model-training-plan.md)。细分节奏、滑条长度、段落条件与多红线仍按独立实验推进。

## 分支管理

| 分支 | 用途 | 合入方式 |
| --- | --- | --- |
| `master` | 可发布的主线，保留已验证功能 | 通过 PR 合入，由维护者审阅；现有 Windows / Ubuntu CI 均须通过 |
| `feat/*`、`fix/*`、`docs/*` | 新功能、修复和文档，每个分支聚焦一件事 | 从最新 `master` 开始，完成后向 `master` 提 PR |
| `exp/*` | 模型训练、新模式、检测方法等实验 | 先提供基线、参数与评估，再把可交付部分拆成 PR |
| `kanzei/*` | 维护者及自动化工作分支 | 遵循同样的 PR 和验证流程 |

协作时不直接向 `master` 推送开发改动，不强推或删除主分支。版本标签和官方 Release 由维护者管理。
外部贡献先 fork，再在自己的仓库建分支提交 PR；新模式单独开发，并验证原有 osu!standard 流程。
GitHub 已启用主分支保护，要求 PR、文档检查与双平台测试通过，禁止强推和删除；使用 squash 合并。
具体步骤和 PR 要求见 [CONTRIBUTING.md](CONTRIBUTING.md#中文)，配置说明见 [仓库规范](docs/governance.md)，实施状态见 [任务队列](docs/TASKS.md)。

## 开源合作协议

欢迎 fork、修复、翻译、模型实验和玩法扩展。项目协作遵循以下约定，代码与模型的使用许可仍以 [LICENSE](LICENSE) 为准：

- 提交贡献时，确认自己有权提供相关代码、模型或素材，并同意原创贡献按本项目现有许可分发；贡献者保留自己的著作权与署名。
- 引用第三方代码、模型和素材时注明来源，保留其许可与版权声明；第三方内容继续适用原有许可。
- 大改动先开 issue 说明目标和方案，或在群里讨论后记录到 issue，避免重复开发。PR 写清改了什么、如何验证和已知限制。
- AI 辅助开发可以使用，提交者负责读懂、检查和验证结果；模型改动提供数据来源、配置与可复现评估。
- fork 和衍生版本请清楚标明与本项目的关系及改动，不冒充官方版本；商业或大规模使用按下方署名条款执行。

完整提交流程见 [贡献指南](CONTRIBUTING.md)。

## 交流与反馈

**QQ 交流群：1124526648**（在 QQ 搜索群号加入）。欢迎交流使用体验、生成谱面、高难度和 highlight 需求，也欢迎参与开发。

可复现的 bug、功能提议和协作任务请同时记录到 [GitHub Issues](https://github.com/kanze1/AUTO-OSU/issues)，
附上软件版本、使用参数、复现步骤及必要日志，方便跟进。

## 许可与署名

代码和模型采用 **MIT 协议加一条署名条款**（见 [LICENSE](LICENSE)）：

- 个人使用、学习、社区里随便玩：只要保留版权声明即可，和普通 MIT 一样。
- **商业使用或大规模部署**（比如公开的网页服务、面向大众分发的应用、批量为平台或社区生成谱面）：
  必须在产品界面、关于页、商店页或文档的显眼位置注明 **AUTO-OSU by kanzei** 并附上本仓库链接 https://github.com/kanze1/AUTO-OSU 。

生成的谱面归你，歌曲归原作者。

## 致谢

- [osu-diffusion](https://github.com/OliBomby/osu-diffusion)（MIT）—— DiT 结构与扩散代码，收录在 `autoosu/ml/coord`。
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)（MIT）—— token 化思路和对比基线。
- [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) —— 训练语料。
- [osu-dreamer](https://github.com/jaswon/osu-dreamer) —— 早期基线。
- [Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC)（OFL）—— 界面字体，以 AUTO-OSU Sans 之名随程序打包。

作者：kanzei
