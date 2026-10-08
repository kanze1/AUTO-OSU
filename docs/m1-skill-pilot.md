# M1 社区正向标签微调试验

**归档决定（2026-10-08）：** 长串保留和技能响应门槛未通过；按维护者决定停止该技能标签发布路线，保留数据、权重和失败证据。新的[模型方案](model-training-plan.md)转向原谱可监督属性与架构评估；这不表示音效、曲线或 new combo 缺少原始监督。

用户在 2026-10-08 要求补齐准备、完成训练并生成不同类型供试听。首轮执行有界的小规模研究试验：使用社区投票标签分别微调节奏和坐标 v0，并生成无标签、跳跃、短串、连打、滑条技巧对照。它不是新默认模型发布；本地独立人工标签复核、完整 M1.3 人评仍待完成。

## 冻结数据与标签

- 标签来源：[project-riz/osu-beatmap-tags](https://huggingface.co/datasets/project-riz/osu-beatmap-tags)，固定 revision `ea7e5b8b39d45ffa4f8d0ee3769024778bf3ea8f`。保留数据卡及文件 SHA-256，使用条件沿用其 osu! terms 标记。
- 使用 `skillset/jumps`、`streams/bursts`、`skillset/streams`、`tech/slider tech` 四项，每项至少两票。这些是社区人类投票，不是自由文本关键词，也不是本项目已完成人工一致性审核的金标准。
- 上游 `tags.csv` 首列叫 `beatmapset_id`，但实际为单谱面 ID；通过 `metadata.csv` 的 `beatmap_id`、mode、所属 set 及文件 MD5 再核对。HF 文本规范化为 LF，因此将其还原成原始 CRLF 序列化再匹配文件哈希，不接受版本错配。
- 缺标签保持未知，只支持正向选择，不提供“排除某技能”。无标签 dropout 为 20%。
- 以音频字节 SHA-256、来源音频身份、beatmapset 和规范化 artist/title 建立传递分组，再确定性分成约 80% / 10% / 10% 的训练、验证、测试集。三个集合的组交集必须为空。
- 这只隔离**本次微调**；v0 已见过的歌曲仍可能在验证或测试中。没有声称独立于 v0 预训练，也没有声称穷尽了所有声学近重复。
- `scripts/prepare_skill_pilot.py` 生成清单、坐标序列缓存、排除记录与来源报告。数据及权重不进 Git。

实际保留 12,832 张谱面；排除 167 张文件版本错配、6 张坐标序列过短或非有限数据。训练 / 验证 / 测试分别为 10,132 / 1,316 / 1,384 张，对应 5,391 / 677 / 712 个歌曲组。训练集中 jumps / bursts / streams / slider tech 的正向记录分别为 5,734 / 4,404 / 2,249 / 877；一张图可以有多个标签。完整清单 SHA-256 和来源哈希保存在 [数据报告](m1-skill-pilot-data-20261008.json)。

## 冻结训练配方

| 设置 | 节奏 | 坐标 |
| --- | --- | --- |
| 基础模型 | 发布的 `rhythm_v0.pt` | 发布的 `coord_v0.pt` |
| 训练方式 | masked 模型微调 | 全 1,000 步噪声过程的 DiT 微调 |
| 技能输入 | 全局条件增加四个正向字段 | 类别条件增加相同四个字段 |
| 初始化 | 新条件列置零，保持 v0 初始函数 | 新条件列置零，保持 v0 初始函数 |
| 物理 GPU | 0、1 | 2、3 |
| 每卡 batch / 序列长度 | 32 / 512 | 32 / 128 |
| 更新步数 | 3,000 | 5,000 |
| 学习率 | 3e-5 | 2e-5 |
| 优化器 | AdamW，weight decay 0.01 | 同左 |
| 调度 | 100 步 warmup，余弦衰减 | 同左 |
| 随机种子 | 1008 | 1008 |
| 验证 | 每 500 步，固定前 16 个 batch | 同左 |

入口为 `python -m autoosu.ml.train_skills`，用 `torchrun` 启动两个双卡任务。训练配方、实际源文件哈希、数据清单哈希、基础模型哈希、逐步损失、初始 / 最佳 / 最后权重全部保存。完整训练与短运行检查使用不同输出目录。

验证分别记录带标签和无标签 loss，固定样本、随机掩码或扩散噪声；每对条件使用相同随机量。最佳权重按两者 loss 之和选取，测试集合不参与选择。非有限 loss 或梯度立即报错退出，不自动重启。

## 试听与验收口径

同歌、同种子、同节奏参数和采样步数，先生成 v0、微调后无标签、单标签四类对照；至少一首额外比较仅节奏、仅坐标、双模型标签。沿用真实星级、结构检查、水印和来源记录，并用独立 parser 读取实际输出。

试听歌曲属于开发展示，不称为全新独立测试。技能响应和实测星级分开报告；不把升星或更密当成技能已经学会。人工试听、客户端试玩和编辑器另存由用户后续验收。

## 2026-10-08 实际结果

两个双卡任务均以退出码 0 完成：节奏 3,000 步、坐标 5,000 步；按冻结验证规则选择的权重分别为第 3,000 / 4,500 步。训练使用账号 `p2522808` 的物理 GPU 0–3，两个完整训练任务均在 07:26:43 UTC 前结束。初始短运行检查与完整训练分开保存。完整[机器可读证据](m1-skill-pilot-evidence-20261008.json)包含配方、源文件哈希、验证记录、权重身份、一次性测试结果及全部试听测量。

| 模型 | 固定 512 张验证图：初始 / 最佳带标签 loss | 全部 1,384 张微调留出测试图：初始 / 最佳带标签 loss | 最佳无标签测试 loss |
| --- | --- | --- | --- |
| 节奏 | 0.443067 / 0.395407 | 0.430342 / 0.388215 | 0.389641 |
| 坐标 | 0.150110 / 0.149932 | 0.154000 / 0.153722 | 0.153737 |

节奏测试损失约下降 9.8%，坐标约下降 0.18%；这是整体微调变化，不能归因于标签。**这里的节奏验证与测试提供了从目标谱面算出的真实局部密度，且只是部分遮罩补全；“无标签”仅表示无技能标签，仍有真实密度，不等同于默认音频生成。** 对于同一个微调模型，加入真实标签相对无标签的损失差很小，而且这些歌曲可能已进入 v0 预训练。测试结果没有用于重新选择权重或采样参数。

实际生成了 4 首开发歌曲 × 6 个版本（v0、新版无标签、四标签），第一首另加 4 个节奏 / 坐标条件消融，共 **28 张 `.osz`**，并导出第一首四标签的点击声试听。冻结 seed=240924、输入星级=6.5、12 次节奏解码、100 次坐标采样、temperature=0.9、CFG=1；参考 timing、关闭 highlight，不做候选筛选或目标星级匹配。推理在本地 RTX 4090 完成。

| 歌曲 | v0 | 新版无标签 | jumps | bursts | streams | slider tech |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joushiki! Butler Koushinkyoku (Cut Ver.) | 5.64 | 5.50 | 5.65 | 5.38 | 5.45 | 5.49 |
| Fruit Salad | 4.42 | 4.66 | 4.38 | 5.05 | 5.07 | 4.33 |
| eyes to eyes | 5.71 | 5.17 | 5.44 | 5.41 | 5.05 | 5.25 |
| Mesheer | 6.72 | 5.87 | 5.81 | 5.64 | 5.66 | 5.83 |

表中是 rosu-pp-py 4.0.2 的实际 NM/stable 星级，版本名称仅表示输入条件。主要发现：

- 第一首 jumps 相对新版无标签的跨度中位数由约 223 增至 235 px；仅坐标标签版本约 245 px。但 Fruit Salad 的 jumps 版本跨度反而下降，效果不稳定。
- Fruit Salad 的 slider tech 版本有 219/230 个滑条，新版无标签为 60/197，说明标签影响物件组成；滑条多不能证明滑条技巧已经学会。
- **长连打未通过。** Mesheer 的 v0 最长四分之一拍连续圆圈为 33 个，新版无标签仅 3 个，streams 标签仅 1 个；所有新版试听图都没有达到 8 个的该类连续圆圈段。第一首消融也未恢复长串。这是保留无标签能力和标签响应的明确反例，不能用平均 loss 下降抵消。
- 四首、单个随机种子、实测星级未匹配只支持开发观察，不能证明同星级技能控制改善；当前保持 v0 默认，不扩大训练或进入 M1.4 发布。后续应先定位长串退化，检验标签与密度、难度的混淆及保留原能力的训练目标，再登记下一轮试验。

实际输出检查：28/28 独立 parser 成功，物件数量一致；没有同时/逆序起点、重叠物件、越界物件头或无效滑条。3,507 条滑条每条采样 1,001 个中心路径点，未发现越界；这不等于精确极值检查或客户端试玩。28/28 与 manifest 和本机生成记录原始字节匹配；26 张检出公开水印，Fruit Salad 的 v0 / slider tech 仅 37 / 11 个圆圈，低于 64 拍点阈值，按规则跳过。水印和本机匹配不认证模型执行。

模型导出时修复了 `torch.__version__` 被序列化为 `TorchVersion` 对象的问题：训练器现将其转为普通字符串；本轮原始权重保留，推理副本只改该元数据，逐张量验证完全相等，并通过 `weights_only=True` 加载。远端测试导出与本地试听导出使用不同 PyTorch 版本，文件字节哈希不同，原始 checkpoint 哈希和张量一致性记录关联二者。

代码验证：186 项测试通过，仓库文档检查通过；没有进行 GUI 标签界面、osu! 客户端试玩或编辑器另存。标签入口目前只在 Python 实验 API 中，现有 CLI / GUI 偏好仍使用 v0 控制逻辑。

## 后续原因诊断与架构调查（2026-10-08）

用户试听后反馈类型区别不明显，要求先调查，不启动新训练。本次只读检查源码，并使用既有权重在固定 256 张验证图上前向诊断，没有更新参数或重新选权重。[诊断原始记录](m1-condition-diagnostics-20261008.json)包含抽样 ID、种子、代码哈希和条件切换结果。此处按有效目标 token 加权，与前面的 batch/map 加权测试平均值不是同一个统计量。

| 节奏前向诊断条件 | 初始模型 loss | 微调模型 loss | 相对下降 |
| --- | ---: | ---: | ---: |
| 提供真实密度、部分遮罩、真实技能标签 | 0.45147 | 0.40773 | 9.69% |
| 不提供真实密度、部分遮罩、真实技能标签 | 0.66798 | 0.65356 | 2.16% |
| 不提供真实密度、全部遮罩、真实技能标签 | 1.06266 | 1.03767 | 2.35% |

在不提供真实密度的部分遮罩设置下，微调模型无技能标签 loss=0.65434、带标签=0.65356，仅相差约 0.12%。全遮罩检查更接近生成的第一次预测，但仍不等于完整迭代解码、手感或风格质量。本次只检查验证集，先前的测试集结果保持原样。

已经确认的设计问题：

- `TickDataset` 在训练时仅以 20% 概率去掉真实密度，验证时从不去掉。`train_skills.evaluate` 只切换技能标签，没有切换密度；最佳权重只按该补全 loss 选择。训练目标缺少完整生成的长串保留和技能响应验收。
- 全谱面的技能标签被直接复制给每个随机窗口，没有标注哪里是连打、哪里是恢复段。抽样 256 张 streams 训练图中，250 张在当前表示下有至少 8 个连续圆圈；2,048 个模拟训练裁剪中有 1,677 个包含该结构。标签不是完全无效，但约 18.1% 的正向裁剪不包含该代理结构。该检查用最后物件后 30 秒代替音频结尾，且是结构代理，不是人工技能标签。
- 节奏模型是约 28.68M 参数、8 层、宽度 512 的 masked Transformer；单 tick 的音频输入为 160 ms 局部 Mel patch，经 MLP 投影，专用 `audio_ctx_layers=0`。主干仍会在窗口内聚合不同 tick 的音频，不能说它只能听 160 ms。技能仅经输入条件 MLP 加到每个位置，没有独立段落计划；这是一种可用但偏弱的条件方案，不是已证明的唯一成因。
- `build_ticks` 只使用四分之一拍网格与六类 token，越过误差阈值的起点会被丢弃，其余会量化；不能忠实表示三连音及更细分节奏。新版没有生成长串不由此单独解释，因为四分之一拍长串本身可以表示。
- 推理时滑条曲线类型与中间控制点数量由 `_slider_structure` 根据长度及随机数决定，SV、组合与音效等还有规则路径。坐标 DiT 获得时间、类型、距离和全局条件，没有直接音频输入；它能改变坐标，但不能自由决定完整滑条拓扑。因此“增加滑条数量”远弱于“学会 slider tech”。
- 两个模型都全参数微调，只使用带正向标签的 10,132 张训练图，没有一般语料回放或保留 v0 输出的约束。实测无标签能力退化已经成立；其中分布偏移、遗忘、条件依赖和迭代解码各自的贡献尚未分离。3,000 / 5,000 步约对应 19 / 32 倍训练图数量的随机裁剪采样，不能仅凭墙钟时间短断言“只要加步数”。

架构资料以作者论文和官方源码为准；读取日期为 2026-10-08：

| 方向 | 已核实内容 | 对本项目的判断 |
| --- | --- | --- |
| [Mapperatorinator 当前架构](https://github.com/OliBomby/Mapperatorinator#model-architecture)及 [V32 配方](https://github.com/OliBomby/Mapperatorinator/blob/main/configs/train/v32.yaml) | 改造的 Whisper 式 encoder-decoder，219M 参数，RoPE、变长 Flash Attention；稀疏事件表示联合生成时间、位置与谱面属性，V32 有坐标细化 token | 最直接的工程参考。可借鉴 tokenizer、条件和联合事件设计；不是把语音预训练权重直接接入，也不能把上游自报速度当作本项目实测 |
| [ITGPT，2026](https://arxiv.org/abs/2607.14148) | DDR/ITG 的分层长上下文音频编码、逐层 FiLM、音频交叉注意力及多步预测 | 段落结构和连续 pattern 训练值得借鉴；其舞步任务与数据不证明 osu!standard 手感 |
| [MuQ，2025 官方实现](https://github.com/tencent-ailab/MuQ) | 提供预训练音乐特征；公开 MuQ 约 300M 参数，权重标注 CC-BY-NC 4.0 | 可作音乐表征研究对照，搭配细粒度 Mel；不先绑定为默认发布依赖，也不预设它能替代精确 onset 特征 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 通过回归速度场训练连续生成模型，已有通用生成实验 | 坐标细化采样效率的后续候选；本身不会补出缺失的技能监督、时值或滑条拓扑。当前坐标模型已经是 DiT/adaLN-Zero，不是“还没用 Transformer” |
| [Rhythm chart discrete diffusion，2026 仓库](https://github.com/qxxiit/rhythm-chart-diffusion) | README 明示 pre-alpha，基线与扩散阶段未完成 | 不作为成熟模型或效果证据 |

建议的下一轮方向是：从头训练新的谱面生成主干作为正式候选，同时保留修正评估后的旧架构作为同预算对照；避免在同样的六类表示上直接投入全量重训。候选采用音频 encoder → 段落条件 → 联合事件 decoder，条件包括技能偏好、目标难度和局部强度，联合输出时值、类型、粗坐标、滑条结构与 SV；必要时再接坐标细化。节奏与空间统一规划，分段控制仍允许恢复段，缺标签仍是未知。

训练数据使用完整、重新按歌曲分组的合法可用语料，并从开始混合有标签与无标签样本；标签样本按歌曲、技能、星级与 BPM 做覆盖和采样检查。模型选择同时检查不提供真实密度的补全、完整多轮生成、无标签能力、同实测星级的技能响应和人工盲评。先把算力用在表示、监督和评估上，RoPE / SDPA 或 Flash Attention / 缓存音频特征是吞吐优化，Flow Matching、Muon 等保留为后续独立消融。四张 48 GB 卡的实际 batch、吞吐和模型规模应经短运行测量后确定，不从上游训练步数推算交付时间。本轮没有启动上述训练或实现新架构。

## 复现与本地产物

训练入口 `python -m autoosu.ml.train_skills` 的全部参数见证据中的 `training.*.recipe.args`。清单与基础权重哈希必须匹配。`scripts/evaluate_skill_models.py` 在冻结权重后对全部微调留出测试图进行一次评估；`scripts/audition_skill_models.py` 生成实际对照图、点击声文件、独立检查和 ZIP。

本次忽略于 Git 的产物路径：`out/m1-training-20261008/` 保存日志、原始 / 推理权重、导出核对和数据报告；`out/m1-skill-audition-20261008/` 保存 28 图、试听说明、点击声音频及 `AUTO-OSU-M1-试听包.zip`。远端原始训练产物在 `/home/p2522808/autoosu/runs/m1_training_20261008/`，固定数据在 `data/m1_pilot_20261008_v3/`。

## English status

The exploratory pilot completed 3,000 rhythm and 5,000 coordinate updates on 10,132 training maps with pinned community-voted positive labels. Missing labels remain unknown. Best checkpoints were frozen before evaluating all 1,384 fine-tuning-held-out test maps and generating 28 development auditions. These splits are not independent of v0 pretraining. The reported rhythm test loss used target-derived oracle density and partial masks, unlike default generation. Post-hoc diagnosis on 256 validation maps reduces the apparent improvement from 9.69% with oracle density to 2.16% without it; switching skill labels then changes loss by only about 0.12%. Long-stream generation regressed: Mesheer's longest quarter-beat circle run fell from 33 in v0 to 3 unlabelled and 1 with the stream label. The fixed six-class quarter-beat representation and rule-selected slider topology also limit expressiveness. This pilot does not pass the skill-control / preservation gate. A joint audio-conditioned event model is recommended for the next controlled study; no new training was started. Independent human review and client playtesting remain pending, and v0 remains the default.
