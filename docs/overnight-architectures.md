# 双模型过夜架构验证

维护者于 2026-10-08 授权节奏模型、空间模型均列出架构并排队验证，接入 W&B。本轮为有界选型，不自动扩大训练、替换默认权重或发布。关联 [M2](https://github.com/kanze1/AUTO-OSU/issues/10)、[M3](https://github.com/kanze1/AUTO-OSU/issues/12)。

## 冻结队列

配置见 [JSON 配方](overnight-architectures-20261008.json)。每候选种子 20261008 / 20261009 各执行一次，共 16 个正式 run。失败记录原因，继续剩余独立项目；不自动重跑失败项目。

| 家族 | 候选 | 要验证的差异 |
|---|---|---|
| 节奏 R0 | `spectral_attributes` | 当前卷积音频、2 层音频上下文＋6 层 masked decoder，作为同预算基线 |
| 节奏 R1 | `attributes` | 线性音频前端，其他深度相同，隔离卷积前端的价值 |
| 节奏 R2 | `spectral_context4_attributes` | 4 层音频上下文＋4 层 masked decoder，固定 Transformer 总深度，比较音乐上下文分配 |
| 节奏 R3 | `spectral_ar_attributes` | 卷积音频、2＋6 层，改为自回归标签预测；比较采样机制，属性训练/推理都使用正确的右移标签 |
| 空间 C1 | `audio-frozen` | 音频残差＋冻结 coord v0，扩大数据后的对照 |
| 空间 C2 | `object-frozen` | 再加入物件归属、真实折返出口、当前带噪相对几何；仍冻结基座 |
| 空间 C3 | `object-joint` | 与 C2 同表示，解冻基座联合微调，检查冻结骨干的限制 |
| 空间 C4 | `spacing-joint` | 在 C3 上加入物件间距分布预测头和监督损失；坐标分支始终接收预测分布，不接收目标间距 |

空间每个 run 还评估旧基座的无距离条件与真实相邻 token 距离条件上界，单列 `oracle_diagnostic`。后者仅诊断已有条件的利用能力，不是可以从音频直接生成的候选，也不是新物件间距接口。

## 数据与预算

节奏沿用原先音频字节、来源身份、集合和规范化歌名的传递歌曲分组：125,230 张训练谱面；验证固定前 128 个不同验证歌曲组。每候选从随机初始化开始，8 层 Transformer、512 宽、8 头，batch 32、512 ticks、6,000 updates；每 1,000 updates 进行自由生成验证。学习率 0.0003、warmup 400、cosine decay，密度以 50% 概率隐藏。验证不提供原谱密度；AR 与 masked 的解码计算量不同，匹配训练样本预算不意味着推理耗时相同。

空间从上述训练/验证清单依次选 4,096 / 128 个不同歌曲组，各取一张谱。可编码部分为 **3,109 / 92 首**，没有用后续歌曲补齐：812 首变速、199 首不足 256 token、12 首滑条编码不支持。原始内容、预处理 mel 和坐标缓存均记录哈希；排除记录留在数据报告中。仅支持恒定 BPM 的结论不能推广到变速歌曲。coord v0 历史训练暴露未知，本轮也不宣称独立泛化。

每空间候选 6,000 updates，每次累积 4 个完整物件片段，每片段最多 256 token。相同种子使用相同歌曲/裁剪选择与训练噪声；分支学习率 0.0001，联合微调基座学习率 0.00001，梯度范数上限 1。验证每歌一个固定完整物件片段，噪声时刻 100/500/900，噪声种子 1949。每 1,000 updates 评估去噪 MSE 和逐物件平均 MSE。

C4 对实际前一物件出口到下一头部的 `log(1 + distance / 512 pixels)` 学习四分量高斯混合分布。第一物件没有已知前驱，不计规划监督；折返出口按奇偶处理。规划 NLL 权重 0.1，坐标网络接收预测分布参数。此处只是连续间距规划，不是方向规划、主观技能分类或跨窗整曲布局；没有固定 jump / stream 距离规则。

测试集不参与架构、种子或 checkpoint 选择。前轮 6 首开发歌小试只作为问题来源，本轮不复用其小样本来下质量结论。

## 指标与完成边界

- 节奏：自由生成 onset F1、密度误差、节奏间隔 JS、滑条比例误差、最长连续圆圈、各属性头参考 NLL/accuracy 和生成匹配计数；W&B 保留逐歌表及两段实际标签序列。所有候选都学习 new combo、头尾音效、折返和滑条结构，不加入已停止的主观技能标签。
- 空间：固定噪声 MSE、物件平均误差、规划 NLL（仅 C4）、基座和 oracle 对照。最佳验证检查点做四段同种子、100 步原始坐标采样，保存坐标表和头部越界数。初始基座也作为 best 候选保留，所有训练 checkpoint 退化时明确显示 `best_step=0`。
- 不以噪声 MSE 或单种子最佳值宣称可玩性提升。比较两个种子相对基线的方向、逐歌回归、稀有属性与输出，再选完整导出和盲测候选；不因某个分数高而自动替换默认权重。
- 节奏采样仍经过已有结构修复；当前记录没有独立的修复前比例指标。空间样例为原始坐标，没有完成滑条拟合、`.osu` 导出、独立解析或客户端试玩。这些仍是后续验收。

## W&B 与队列

[W&B 项目](https://wandb.ai/kanzei/autoosu-architecture)，正式 group 为 `overnight-20261008`，工程短测为 `overnight-20261008-preflight`。使用在线模式，保存 run ID 和 URL，记录源配置/哈希、梯度、loss、逐歌验证与样例。上传数值和派生样例，不上传原歌曲和训练权重。在线初始化失败会报错，不静默改成 offline。

GPU 0–1 执行节奏队列，2–3 执行空间队列，每卡同时一个候选；使用 UUID 固定物理设备。调度器启动时核实空闲并建立本账号协作锁，逐项调度前检查占用。不会接管其他进程。SIGTERM 会终止该队列自己的子进程组；不会安排自动恢复。

运行位置为 `/home/p2522808/autoosu/runs/overnight_20261008/`。`queue/dispatch.json` 记录控制器和子进程 PID、账号、GPU、状态、退出码和 W&B 链接；每项有 stdout 日志、配方、验证 JSON、best 和恢复状态。结束后生成 `queue/comparison.json`。检查点用于复查/恢复，节奏 DataLoader 的预取状态不保证逐 bit 重放。

## 复现入口

先用 `scripts/prepare_coord_experiments.py` 从冻结清单与源分片提取数据，输出目录必须不存在。随后：

```bash
python scripts/make_experiment_queue.py --base /home/p2522808/autoosu --source <source-checkout> --run <run-root> --recipe docs/overnight-architectures-20261008.json --coord-base <coord_v0.pt> --coord-data <prepared-coordinate-data>
python scripts/run_experiment_queue.py --manifest <run-root>/queue-manifest.json --out <run-root>/queue
```

源版本记录在部署目录 `source-revision.txt`，每个 run 另有源文件哈希。manifest 和输出目录拒绝覆盖，已有队列不能因重复命令被当作新任务重新执行。计划完成不等于训练完成；实际执行状态见 `dispatch.json`。

## 启动验收

[启动快照](overnight-launch-20261008.json)确认北京时间 2026-10-08 22:31 起正式队列正在执行：账号 p2522808、控制器 PID 4161825，首批四个训练 PID 4161831 / 4161835 / 4161839 / 4161843 分别使用 GPU 0–3；W&B 服务端确认在线运行且收到更新。其余 12 项等待自动调度。八种路径的短程训练和生成均完成，226 项测试与仓库检查通过。部署源码为 `fa833aa`。这只是启动验收，尚未完成 16 项对照，也未进行发布验收。
