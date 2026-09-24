# 难度与 highlight 控制：验收版

源码版本 **0.3.0.dev3**，权重仍为 v0。新功能位于高级选项中的「难度与 highlight 控制（实验）」。默认留空时沿用原有生成方式；本轮增加控制接口和有限候选搜索，不声称模型已经稳定达到指定星级或理解音乐高潮。

## 可操作的功能

| 控制 | 行为 |
| --- | --- |
| 目标实测星级 | 最多生成 1–3 个候选，实际测量后选取最接近目标且通过结构 / 峰值检查的结果；未达到 ±0.5★ 时明确显示 |
| 密度条件 | 常数 0–16 物件/小节，或 JSON 中的逐段曲线；使用训练时前后 8 小节平滑和边缘归一化 |
| 跨度倍率 | 0.5–1.5；先无距离条件生成一次参考位置，再以逐 token 像素距离做第二次采样，只缩放圆圈 / 滑条头的进入距离 |
| 手动 highlight | 原音频秒数、强度、进入与回落时间；统一驱动密度、跨度、combo、音效和 kiai |
| 自动 highlight | 按小节综合响度、打击与和声 onset 的对比提出最多两个区间；低对比或太短则不添加 |
| 独立 SV 开关 | 新 highlight 默认不额外加速滑条；可另行启用预设 kiai SV |
| 计划导入 / 导出 | GUI、CLI、独立 GPU worker 与文件夹批量使用相同 JSON；超出某首歌的手动区间明确报错 |

跨度倍率 1.0 也会启用第二次采样，与“留空、无距离条件”不同。候选筛选优先排除重叠、越界物件头、无效滑条、同时开始和孤立 strain 峰值；锚点在场外不等于曲线路径出界，不将其直接作为拒绝条件。没有合格候选时保留首个结果并记录 `fallback_unverified`，目标不算达成。

## 操作

先固定歌曲、BPM / offset、随机种子与 Insane，再比较原版和一个新控制。验收时首选实测目标 6.5★；输入范围 1–12 不代表整个范围都已通过质量验证。

```powershell
python -m autoosu "song.mp3" -d Insane --seed 240924 --target-star 6.5 --candidates 3
python -m autoosu "song.mp3" -d Insane --seed 240924 --spacing-scale 1.2
python -m autoosu "song.mp3" -d Insane --highlight 30:55:1 --highlight 80:100:0.7
python -m autoosu "song.mp3" -d Insane --highlight-mode auto
python -m autoosu "song.mp3" -d Insane --control-plan "plan.json"
```

计划示例（秒数对应原音频，不需要自行减去 osu! 导出偏移）：

```json
{
  "schema": "autoosu.control-plan/1",
  "controls": {
    "target_stars": 6.5,
    "candidates": 3,
    "highlight_mode": "manual",
    "highlights": [
      {"start_s": 30, "end_s": 55, "strength": 1, "attack_s": 2, "release_s": 4}
    ],
    "highlight_sv": false
  }
}
```

常数 `density` 与 `density_curve` 二选一，曲线为递增时间的 `[秒数, 物件/小节]` 数组，例如 `[[0,6],[30,12],[60,6]]`。曲线首末值延伸到歌曲两端。`highlight_density` / `highlight_spacing` 默认 true，可在实验计划中单独关闭模型条件以做消融。它们不关闭 combo、音效或 kiai。

手动核心区间不能重叠，进入 / 回落采用最大包络合并。默认密度增益为 `1 + 0.35 × 包络`，进入距离增益为 `1 + 0.2 × 包络`。未知密度时先采样节奏，按物件起点构造本曲密度，再加入段落增益；不会把未知条件直接当成固定的全曲高密度。训练标签约覆盖前后 8 小节，因此短爆点不会成为未经训练的逐秒密度尖峰。

每张谱面的生成清单、`.evaluation.json` 与 worker / 批量报告记录请求设置、实际 highlight 区间、候选参数与实测指标、拒绝原因和选中的候选。结果日志显示目标是否达到和 highlight 秒数；主界面显示当前启用的控制，避免隐藏设置影响下一首歌。

## 验证与剩余验收

开发对照使用预先冻结的 8 首歌曲，参考 timing、同一随机种子、v0 权重和固定采样设置。自动区间尚无人工位置标签；手动消融使用固定的歌曲中部范围，仅验证控制响应，不把它当作音乐高潮标注。冻结候选方案后才使用 24 首保留歌曲，不根据保留集结果继续挑参数。

完整记录见[参数、代码 / 权重哈希、逐歌响应与失败样本](generation-controls-evaluation.json)。先在开发集做 40 张密度 / 条件对照和 64 张控制消融；候选搜索比较了四种开发策略，最后冻结为：首个候选使用原始条件，第二个按实测差值修正星级条件，第三个保留前两个中更接近目标的合格节奏，只按 `(目标 / 实测)²` 调整距离，倍率限制在 0.5–1.5。不会同时强制增加密度。全部候选最多三个，任一合格结果进入 ±0.5★ 即停止。

| 固定目标 6.5★ | 达到 ±0.5★ | 比例 | 实测星级中位数 |
| --- | --- | --- | --- |
| 开发集，参考 timing | 7/8 | 87.5% | 6.24★ |
| 预留集，参考 timing | 21/24 | 87.5% | 6.22★ |
| 预留集，自动 timing | 19/24 | 79.2% | 6.13★ |

自动 timing 尚未达到 80% 门槛，且尚无人工手感结论，因此保留实验状态。参考 timing 的三首失败结果为 5.33 / 5.94 / 5.79★；自动 timing 的五首为 5.27 / 5.82 / 5.62 / 5.78 / 5.42★，对应歌曲哈希和候选明细均保留在报告中。预留歌曲已使用，后续改进需要新的独立验收集；当前分组只保证开发与预留歌曲分离，不保证与模型训练集分离。

其他对照结果：

- 8/8 首无新控制的输出与归档 `2d2e0bd` 原版逐字节相同。
- 跨度 0.85 / 1.0 / 1.2 在 8/8 首上呈现实测星级递增；它仍是模型条件，不能承诺任意歌曲都严格单调。
- 手动中部区间在 8/8 首上提高区间 NPS；同样节奏上再加跨度控制，8/8 首的区间 spacing 中位数高于“只控制密度”。这不等于比原版更好玩；例如加密后每次跳跃可能比稀疏原版更短。报告同时列出区间外 NPS、全曲诊断和局部 strain。
- 自动 highlight 在开发集 6/8 首、预留集 14/24 首提出区间，其余不添加。没有人工位置标签，不报告音乐高潮定位准确率。
- 64 张开发消融、8 张最终开发目标图和 72 张预留输出，共 144 张通过独立 `slider 0.8.2` 解析；没有重叠、同时起点、越界物件头或无效滑条字段。对 62 条场外锚点滑条各采样 1,001 个曲线点，未发现中心路径越界；这项采样不代替客户端检查。

软件验证：149 项测试通过；独立 CUDA worker 完成手动区间 + 目标星级单曲，以及两首自动 highlight 批量；请求、JSON 结果、导出清单、实测报告和本地来源记录一致。另以一短一长合成歌曲检查手动区间越界：短歌明确失败，批量继续完成长歌。真实中英文窗口的表单应用、设置保存和实际 worker 结果日志已检查：[中文](images/controls-zh.png)、[English](images/controls-en.png)。

本地 PyInstaller 验收包已构建，附带 v0 权重；独立 EXE 实际完成 CPU 双模型生成、控制计划导入和来源检查，其内置 GPU 安装源包的 45 个 Python 文件与构建源码一致。EXE 主窗口也已打开检查。合成歌曲的 EXE 检查用 10 次坐标采样，只证明打包功能正常；真实歌曲实验均用 100 次。

本地交付目录为 `out/acceptance-0.3.0.dev3/`，包含独立应用、已有 GPU 环境的快捷入口、4 首 / 16 张对照谱面、验收说明和空白反馈表。对照包仅修改标题、作者栏和难度名以避免导入相互覆盖，物件内容指纹与实测星级保持一致；另附副本清单，不沿用失效的原始文件哈希声明。部分原版对照圆圈不足，正确保留“水印信息不足”，不伪造检出。

尚未进行 osu! 客户端导入、编辑器另存和人工试玩。用户负责这一步手感与音乐性验收；局部重生成、细网格训练、多红线和更换模型仍按后续任务推进。

复现入口（歌曲与权重保留在本地，不提交 Git）：

```powershell
python scripts/evaluate_density_response.py --help
python scripts/evaluate_controls.py run --out out/new-development --partition development
# 先冻结候选和阈值，再使用独立预留歌曲。每种 timing 使用单独目录。
python scripts/evaluate_controls.py run --out out/new-reference --partition heldout
python scripts/evaluate_controls.py run --out out/new-automatic --partition heldout --timing automatic --variants target65
python scripts/evaluate_controls.py report --out out/new-reference --partition heldout
# 独立 parser 环境需要 slider。
python scripts/evaluate_controls.py curves --out out/new-reference --partition heldout
```

脚本核对歌曲哈希、记录推理代码和模型哈希；目录中已有不同配方时拒绝混写。密度对照使用归档代码，控制消融与最终候选各自保留实际执行配方。

## English

The 0.3.0.dev3 development app adds optional measured-star candidate selection, smoothed local density curves, two-pass pixel-distance conditioning, manual audio-time highlights and conservative automatic proposals. Defaults preserve the original generator. Target selection is bounded to three candidates and explicitly reports failure; neither measured stars nor parser validity establish playability.

Manual regions coordinate density, spacing, combos, hitsounds and kiai, with optional slider-velocity changes. Automatic locations combine bar-level loudness and percussive/harmonic onset contrasts, abstaining on insufficient contrast. Plans are shared across the GUI, CLI, isolated worker and folder batches. Original audio seconds are converted to map time only on export.

Reports retain candidate measurements, rejection reasons, selected settings and observed region metrics. Development uses the existing eight-song split, followed by one frozen evaluation on the 24 reserved songs. Automatic locations are unlabelled proposals; human listening, osu! editor round-trips and playtesting are acceptance steps, not inferred from automated tests. Local regeneration and model retraining remain follow-up work.

With a 6.5-star target, final development met +/-0.5 on 7/8 songs; reserved evaluation met it on 21/24 with reference timing and 19/24 with automatic timing. The latter is below the 80% gate, so the controls remain experimental. There were no recorded structural failures across 144 independently parsed outputs, and 62 flagged-anchor slider curves had no out-of-field samples. Manual controls increased central NPS on 8/8 development songs; adding distance conditioning increased central spacing over density-only controls on all eight. Automatic proposals were present on 14/24 reserved songs, with no human location-accuracy claim. The [machine-readable report](generation-controls-evaluation.json) retains settings, hashes, per-song results and all misses.

149 tests, actual isolated single/batch workers, bilingual UI checks and a full local Windows executable smoke test passed. The private acceptance folder contains four songs with four metadata-separated versions each; maps, audio and model binaries remain outside Git. No GitHub release, osu! client playtest or editor resave is claimed. New independent songs are required for evaluating future tuned policies.
