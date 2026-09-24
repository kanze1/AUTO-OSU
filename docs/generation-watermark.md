# 新图内容水印

源码开发版 **0.3.0.dev2** 默认在新生成的 osu!standard 谱面中加入实验性内容水印。GUI 的「检查谱面来源」和 CLI 的 `--check-source` 均可读取。已发布的 0.2.0 下载尚不包含此功能，模型权重仍为 v0。

**旧图不做追溯归因。** 旧版分类器实验保留为历史记录，不再推进。此次读取历史谱面仅为测试“未加水印会不会误报”，不判断它们由谁制作。

## 使用与结果

正常生成即可，无需额外开关。单曲、文件夹批量、规则引擎和两种模型路径都走同一个导出步骤。生成日志显示水印是否成功加入；生成记录、worker / 批量结果和难度报告也保留这一状态。

```powershell
python -m autoosu --check-source "map.osz" --source-report "report.json"
python -m autoosu --check-source "D:\Beatmaps" --recursive
```

检查不需要模型、原始歌曲、网络或生成者的本机记录。删去 Tags、改标题、移除谱包中的 JSON 后，仍可检查坐标里的标记。报告分开显示水印、文件声明和本机匹配；本机记录匹配优先显示，但不会隐藏水印结果。

水印是**公开来源标记**，不是签名或模型执行证明。它可以被复制、主动移除，也可以由 fork 跳过；它声明的“双模型 / 节奏模型 / 坐标模型 / 纯规则”同样不构成认证。确切权重哈希仍由生成清单声明。未检出只表示没有足够标记，不能推断人工制作。没有在客户端内置秘密密钥或隐蔽执行入口。

公开实现可以绕过生成后水印，这也是 [NIST AI 100-4](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-4.pdf) 第 18–20 页讨论的限制；以下实测只支持此版本、此测试范围的保留率。

## 固定协议

实现：[autoosu/watermark.py](../autoosu/watermark.py)，版本 `autoosu-circle-parity/1`。

- 仅调整圆圈头坐标，每个轴最多 1 像素，最大位移 √2 像素。保持场地边界，并保留原本已有的圆圈半径余量。滑条、锚点、物件时间、音效、timing、kiai、物件数量均不改变。
- 按最早红线和每拍 4 tick 计算物件所属拍点。公开 SHA-256 码本由协议版本、引擎类别和 tick 决定圆圈 x/y 奇偶性；不使用元数据或歌曲名称。
- 同 tick 只贡献两位，重复物件不能放大证据。互相冲突的位保留在分母中但不计匹配。最多使用前 8,192 个不同 tick。
- 至少需要 64 个不同 tick 上的圆圈，即 128 位；匹配率至少 85%。四种引擎、四种整体坐标奇偶偏移，共 16 个固定候选。恰好一种引擎通过才检出，多种通过则弃判。分数不是来源概率。
- 圆圈不足时不强塞物件，保持谱面不变并显示 `insufficient`；格式或数值不支持时显示 `unsupported`。短谱和大量滑条的谱面可能没有足够容量。
- 在最终星级测量、内容指纹和导出之前加入水印，因此记录与星级对应实际输出。没有后续偷偷改包的步骤。

改变红线相位、重新吸附坐标、集中改写圆圈或主动清除奇偶性，均可能破坏水印。局部片段扫描、任意变速后的恢复、恶意去水印不在此版能力范围。

## 固定验证结果

完整计数、文件哈希和逐图结果：[watermark-evaluation.json](watermark-evaluation.json)。复跑脚本：[scripts/evaluate_watermark.py](../scripts/evaluate_watermark.py)。原歌曲、谱面与生成产物留在忽略目录，不提交仓库。

先固定码本、64 tick 下限与 85% 阈值，再读取独立对照的分数；本次没有按测试集调阈值。新增对照按音频哈希去重，排除之前旧版分类器用过的全部歌曲，以及 Q1 的全部 32 首歌曲；Q1 的 24 首保留集仍未使用。没有按水印分数挑选对照。

| 检查 | 结果 |
| --- | --- |
| 新增未加水印对照，1,650 首独立音频 | 1,504 张可检查，0 次误报；127 张圆圈不足，19 张结构不支持 |
| 可检查对照的最高匹配率 | 62.86%，固定门槛为 85% |
| 40 首真实歌曲 × 四种引擎的 160 张基线输出，加水印前 | 156 张未检出，4 张容量不足 |
| 同一组输出，加水印后 | 156 张检出，4 张容量不足；未混淆引擎声明 |
| 删全部元数据，改换行并加 UTF-8 BOM | 156 张检出，4 张容量不足 |
| 每 10 个圆圈改一个：x +7、y −5、时间 +3 ms | 156 张检出，4 张容量不足 |
| 每 10 个圆圈删除一个 | 150 张检出，10 张容量不足；其中 6 张从可检查降为不足 |
| slider 0.8.2 独立解析并重新序列化 160 张谱面 | 156 张检出，4 张容量不足 |
| 同图加水印前后实测星级，rosu-pp-py 4.0.2 / NM stable | 最大绝对变化 0.010613★，平均绝对变化 0.002222★ |

零误报不代表真实误报为零。以 1,504 张可检查对照作为独立样本估计，一侧 95% 上界约 **0.199%**，尚不能证明低于 0.1%；同歌不同录音、谱师与风格相关性仍可能影响推广。这是公开标记的初步验证，不能作为自动处罚的依据。

160 张基线为冻结代码 `8f50892`、v0 权重、种子 240924、星级条件 5.5、节奏 12 步、坐标 100 步、temperature 0.9、CFG 1.0 的真实输出。这里只在已生成的固定内容上加标记做前后对照，不重训、不替换模型；规则模式没有消费模型星级条件。脚本逐字段确认除圆圈 x/y 外没有其他内容变化。

此外，用当前开发代码独立启动实际 worker：同一首歌的 Hard / Insane，双模型、两种混合、纯规则共 8 张；另做两首歌的文件夹批量，共 2 张。模型路径使用 RTX 4090 / CUDA，纯规则使用 CPU。**10 张全部加入并检出水印**，生成清单、实际输出哈希、测量报告和本机记录一致；移除标签与 JSON、重新打包、换成空记录库后仍检出。CLI JSON 导出与中英文来源检查界面均已验证。

本地完整测试 **129 项通过**。GUI 截图：[中文](images/watermark-zh.png)、[English](images/watermark-en.png)。

## 仍待验证

osu! stable / lazer 的实际导入、编辑器另存和人工试玩尚未完成；独立解析器另存不能替代这些验证。当前也没有重打 Windows 发布包。后续按任务表推进高难度控制和 highlight 时，继续检查水印对局部 strain、堆叠和手感的影响；小星级变化不等于手感完全不变。

## Reproduction

With the frozen baseline corpus and the eight local dataset shards available:

```powershell
python scripts/evaluate_watermark.py prepare
python scripts/evaluate_watermark.py run
python scripts/evaluate_watermark.py roundtrip
```

`prepare` refuses to overwrite a recipe; `run` checks the frozen codec and input hashes. `roundtrip` requires the optional `slider==0.8.2` parser. The star comparison requires the pinned `rosu-pp-py==4.0.2` (Python 3.11+). Songs and maps stay under `out/watermark-evaluation`; the checker itself has no ML or third-party parser dependency.

This development version marks new outputs only. The public marker is removable and forgeable, its engine label is a claim, and absence is inconclusive. The fixed evaluation found zero false positives in 1,504 eligible independent unmarked controls, with 146 abstentions; its one-sided 95% upper bound is approximately 0.199%, not proof of a sub-0.1% false-positive rate. All 156 eligible marked maps survived the specified small edits and independent parser serialization. Actual osu! client resaving and human playtesting remain unverified.
