# Contributing to AUTO-OSU

[中文](#中文) · [English](#english)

## 中文

欢迎参与代码、文档、翻译、模型实验和新模式开发。QQ 交流群：**1124526648**；需要持续跟进的任务请记录到 [GitHub Issues](https://github.com/kanze1/AUTO-OSU/issues)。

### 分支与提交

1. 外部贡献先 fork 仓库，从最新 `master` 创建功能分支。`feat/*` 用于功能，`fix/*` 用于修复，`docs/*` 用于文档；`exp/*` 用于训练与探索。维护者和自动化工作可用 `kanzei/*`。
2. 一次 PR 聚焦一件事。新模式、模型更换和大规模重构先开 issue 讨论目标、接口与验证方法，实验分支整理后再提交可合入的改动。
3. 向本仓库 `master` 提交 PR，由维护者审阅、合并。不要直接推送开发改动到主分支，不强推或删除 `master`；官方标签与 Release 由维护者发布。
4. 合并前保持分支可与最新主线合并，并通过现有 Windows / Ubuntu CI。修改共享生成、批量处理或导出流程时，同时验证原有 osu!standard 行为。

### PR 需要说明什么

- 解决的问题、关联 issue、最终行为，以及复现或使用示例。
- 实际运行的检查和结果；没有运行的验证直接注明。界面改动附截图，中英文界面与 README 同步更新。
- 模型或生成策略改动附代码/权重版本、数据来源、固定样本、种子、关键参数和对照结果；区分离线指标、端到端生成与人工试玩。
- 检测器改动报告独立测试上的误报、召回与适用范围，不将“未检出”标成“人工制作”。

本地开发的基本检查：

```bash
pip install -e ".[dev]"
python -m pytest -q
git diff --check
```

纯文档改动检查内容、链接、格式与中英文一致性即可；涉及生成器的改动还需要有代表性的真实歌曲验证，不能只凭规则模式单元测试判断 AI 质量。
模型权重通过发布资源分发；不要将虚拟环境、训练语料、歌曲、生成结果或大型 checkpoint 提交到 Git。必要测试素材应有清楚的来源与使用权限。

### 开源合作协议

项目现有许可是 [MIT 加署名条件](LICENSE)，同时覆盖项目发布的模型。提交原创贡献表示你有权提交，并同意按这一许可分发；著作权和贡献署名由贡献者保留。引入第三方内容时注明来源，保留原许可，说明其如何与项目一起分发。

欢迎独立维护 fork，也欢迎将改进回馈上游。衍生版本注明自己的维护者、改动与模型来源，不冒充官方版本；商业或大规模使用遵守 LICENSE 的 **AUTO-OSU by kanzei** 署名要求。模型、代码与素材的权利各自处理，项目许可不授予第三方歌曲或素材的权利。

允许 AI 辅助开发，提交者仍负责代码质量、来源说明和真实的验证结果。重要方案与决定留在 issue / PR 中，方便后来参与的人理解。合并依据是改动质量、项目方向和验证结果；贡献不自动意味着合并或发布。

使用项目生成并分享谱面时，明确注明生成来源。osu! 的 Ranked 要求参见 [官方 AI policy](https://osu.ppy.sh/wiki/en/Ranking_criteria#ai-policy)。

## English

Code, documentation, translations, model experiments, and new game modes are welcome. **QQ group: 1124526648**. Track ongoing work in [GitHub Issues](https://github.com/kanze1/AUTO-OSU/issues).

### Branches and submission

1. External contributors should fork the repository and branch from the latest `master`. Use `feat/*` for features, `fix/*` for fixes, `docs/*` for documentation, and `exp/*` for training or exploration. Maintainer and automation work may use `kanzei/*`.
2. Keep each PR focused. Discuss the goals, interfaces, and validation of a new game mode, model replacement, or major refactor in an issue first. Extract mergeable changes from experiments before submitting them.
3. Open a PR against this repository's `master` for maintainer review and merging. Do not push development changes directly to the main branch, force-push it, or delete it. Maintainers publish official tags and Releases.
4. Keep the branch mergeable with the latest mainline and pass the existing Windows / Ubuntu CI before merging. Changes to shared generation, batch, or export code must also validate existing osu!standard behavior.

### What to include in a PR

- The problem, linked issue, resulting behavior, and a reproduction or usage example.
- Checks actually run and their results; state any checks not run. Include screenshots for UI changes and update both interface languages and READMEs together.
- For model or generation changes: code / weight versions, data sources, fixed samples, seeds, key parameters, and comparisons. Distinguish offline metrics, end-to-end generation, and human playtesting.
- For detectors: false-positive rates, recall, and supported scope on independent test data. Do not label "no detection" as "human-made".

Basic local checks:

```bash
pip install -e ".[dev]"
python -m pytest -q
git diff --check
```

For documentation-only changes, check content, links, formatting, and bilingual consistency. Generator changes also need representative real-song validation; rule-based unit tests alone do not measure AI quality.
Distribute model weights through release assets. Keep virtual environments, training corpora, songs, generated maps, and large checkpoints out of Git. Any necessary test assets need a clear source and permission to use them.

### Open-source collaboration agreement

The existing project licence is [MIT with an attribution condition](LICENSE), including models released by the project. Submitting an original contribution means you have the right to submit it and agree to distribute it under that licence; contributors retain copyright and credit. Identify third-party sources, preserve their original licences, and explain how they can be distributed with the project.

Independent forks and upstream contributions are welcome. Identify the maintainer, changes, and model sources of derivative versions, and do not present them as official releases. Commercial or large-scale use must meet the **AUTO-OSU by kanzei** attribution requirement in LICENSE. Rights to models, code, and assets remain distinct; the project licence grants no rights to third-party songs or assets.

AI-assisted development is welcome. Submitters remain responsible for code quality, source attribution, and accurate validation reports. Keep significant proposals and decisions in issues / PRs for future contributors. Merging depends on quality, project direction, and validation; submission does not guarantee a merge or release.

When sharing generated beatmaps, disclose their source. See the official osu! [AI policy](https://osu.ppy.sh/wiki/en/Ranking_criteria#ai-policy) for Ranked requirements.
