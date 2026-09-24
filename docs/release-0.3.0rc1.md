# 0.3.0rc1：默认自动 highlight 与高难度验收

本轮保持 v0 模型，先作为候选版交付。新配置默认自动 highlight；用户可选择手动、关闭或原有 kiai，已有配置中明确保存的模式保留。目标星级仍按实际计算结果显示，未达到时明确提示。

## 维护者反馈

维护者检查了 0.3.0.dev3 验收包前三首的 A/B/C/D 四种版本，反馈整体比之前略有改善、比较自然，没有发现明显不合理的地方。第一首最喜欢 C（Manual highlight）；第二首 A（原版）的滑条略多。第四首未纳入这次已确认的反馈。

这说明手动段落控制值得保留，也支持把自动模式作为易用的默认入口。没有据此调整滑条比例或重新训练模型，避免混入尚未对照过的改动；滑条偏多已记录为后续诊断。反馈没有明确说明是否完成手动打图或编辑器另存，因此仅记为人工检查，不替代这两项记录。

## 高难度样例

沿用四首已知开发歌曲，固定参考 BPM / offset、种子 240924、12 次节奏采样、100 次坐标采样和 v0 权重，生成目标 7 / 8 / 9★ 的自动模式；第一首额外生成同三档手动模式，沿用之前喜欢的歌曲 35%–65% 区间。共 15 张。它们用于这次试听 / 试玩，不作为新的独立测试集。

每个难度名同时写明目标和实测星级。候选数量最多三个，选取结果的结构诊断和所有候选指标保留。打包对照仅改元数据，内容指纹和实测星级必须一致。

```powershell
python scripts/generate_difficulty_ladder.py generate
python scripts/generate_difficulty_ladder.py package
```

实测结果如下（rosu-pp-py 4.0.2，stable / NM）：

| 歌曲 / 模式 | 目标 7★ 实测 | 目标 8★ 实测 | 目标 9★ 实测 |
| --- | --- | --- | --- |
| Joushiki! Butler Koushinkyoku (Cut Ver.) / 自动 | 6.60★ | 7.67★ | 8.72★ |
| 同曲 / 手动 | 6.75★ | 7.74★ | 8.61★ |
| Fruit Salad / 自动 | 6.68★ | 6.59★ | 6.26★ |
| eyes to eyes / 自动 | 6.30★ | 6.04★ | 5.61★ |
| Mesheer / 自动 | 6.88★ | 7.80★ | 8.81★ |

15 张中 10 张在目标 ±0.5★ 内。第二、三首提高目标后未继续升星，说明现有条件响应存在明显的逐歌差异；保留全部失败样例，不将这些已知歌曲计为独立达标率。原始输出、全部候选、固定参数和代码 / 权重哈希对应的精简记录见 [JSON 报告](high-difficulty-rc1.json)。

15 张均通过结构检查，没有重叠物件、同刻 / 倒序起点、越界物件头 / 锚点或无效滑条；独立 `slider 0.8.2` 解析通过，每条滑条采样 1001 点未发现越界路径（未模拟 stacking）。重新打包后内容指纹和实测星级不变，15 张内容水印仍检出。新的高难度样例尚待维护者检查 / 试玩。

## 默认行为检查

- 150 项本地测试通过，包括默认 auto、显式历史配置和 CLI 覆盖的兼容性检查。
- 独立 CUDA worker：单曲四档与两首文件夹批量共 6 张，不传 controls 时均使用 auto；额外显式 legacy 的输出与之前冻结的基线字节一致。
- 上述 7 张均完成实际双模型推理、实测星级及来源检查（本机记录匹配），结构检查未发现异常。
- 中英文真实 GUI 均检查了默认 auto、四种模式切换及应用后的主界面状态：[中文](images/controls-default-zh.png)、[English](images/controls-default-en.png)。

Windows 候选包的构建、冻结程序启动和推理结果由发行包附带的 `verification.json` 记录；这些结果与 osu! 客户端手感验收分别处理。

## 发布范围

- 新图内容水印、来源检查、本机记录与实测星级。
- 默认自动 highlight，手动 / 关闭 / 原有 kiai 模式可选。
- 密度曲线、跨度条件和有限目标星级筛选。
- Windows 应用保留 CPU 路径与独立 GPU 配置。升级后 GPU 环境需要匹配应用版本。
- 高难度控制与自动段落位置保留实验标记。之前自动 timing 的 6.5★ 独立评估为 19/24，不能用这次已知歌曲样例替换该结果。

发布包不包含第三方歌曲或验收谱面；试玩包仅保留在本地。当前公开稳定版仍为 0.2.0，本轮准备候选发布包。

## English

0.3.0rc1 makes automatic highlights the default for new configurations, retaining manual, off and original-kiai modes and preserving saved choices. The v0 model weights are unchanged. The maintainer reviewed the first three A/B/C/D comparison sets and found them generally natural, preferring the first song's manual version and noting excess sliders in the second song's original version. This is recorded as a qualitative review, not an editor-resave or confirmed manual-playtest result.

A fixed 15-map audition set requests 7/8/9 stars on four previously used development songs, with three additional manual versions of the first song. Ten maps are within half a star of their requested rating; two songs do not get harder as requested. Difficulty names distinguish requested and measured ratings. All 15 pass structural and independent parser checks, and their watermarks survive metadata-only repackaging. These examples do not replace independent evaluation or human gameplay acceptance.

Local validation includes 150 tests, real CUDA workers for all four presets and a two-song folder batch using the default, a byte-identical explicit legacy regression, and both interface languages. The application is prepared as a release candidate; automatic location and high-star targeting remain experimental, and third-party audio/maps are excluded from the public application package. Frozen Windows build and executable checks are recorded separately in the release asset's `verification.json`.
