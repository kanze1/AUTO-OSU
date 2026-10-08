# M1 标签训练准备审计

2026-10-08 开始执行 M1.1。当前阶段是语料盘点和标签覆盖审计，**M1.2 标签条件训练尚未启动，新权重尚未产生**。本次不会把旧 v0 训练入口重新运行当成标签训练。

## 全量结果

74 个分片扫描完成，退出码为 0；[原始统计报告](m1-data-audit-20261008.json)保留输入索引、脚本和各分片元数据摘要。

| 检查 | 结果 |
| --- | --- |
| 预处理谱面 / 原始记录连接 | 139,582 / 139,582；缺失 0 |
| 预处理音频条目 / beatmapset | 32,578 / 36,204 |
| 非空自由文本标签 | 135,858 张 |
| 含技能相关独立词项 | 1,815 张；只是检索候选，不是合格训练标签 |
| 已复核技能标签 | 0 张；所有技能值保留未知 |
| 人工抽检队列 | 95 张，各自来自不同的来源音频身份 |
| 旧节奏训练实际加载 | 134,692 张训练 / 4,220 张验证；至少 50 物件 |
| 同一预处理音频跨训练 / 验证 | 182 个音频条目，共 1,846 张图，其中 721 张在验证集 |
| 6–7★ / 7–8★ / 8–9★ / 9–10★ | 10,518 / 2,759 / 687 / 179 张；沿用索引星级，未重新计算 |

这证明仅按 beatmapset 划分不足以隔离此语料中的音频；受影响验证图占旧验证集约 17.1%。没有因此推算旧模型性能的下降幅度。来源音频哈希分组得到了相同交叉计数，但它仍是来源声明，不替代音频字节和近重复核对。

另用实际 `autoosu.ml.dataset.load_index` 独立加载训练 / 验证列表并求音频交集，复核了 182 / 721 的结果；逐行核对清单唯一性、139,582 条计数、标签全为未知、95 张抽检音频不重复及脚本哈希，全部通过。已有 v0 训练覆盖与新测试独立性仍须分别处理。

## 可复现入口

```bash
python scripts/audit_training_data.py \
  --shards data/hf/compressed \
  --index data/prep/index.jsonl \
  --out runs/m1_audit_20261008/evidence
```

输出目录必须不存在，避免覆盖人工复核记录。程序读取已有数据，不改动训练集、旧 checkpoint 或运行中的其他任务。输出包括：

- `manifest.jsonl`：预处理条目与原始分片的精确连接、原始自由文本标签、谱面内容哈希、来源声明的音频哈希。
- `report.json`：完整扫描状态、来源连接缺失、星级覆盖、关键词覆盖、旧划分中的同音频交叉情况、输入与脚本哈希。
- `review_queue.jsonl`：按星级段和关键词命中分层的确定性人工抽检队列；同一来源音频身份不重复入选，每层最多五张。
- `missing.json`：未连接到原始记录的 `(beatmap_id, track)`；非空时程序以失败状态退出。

完整清单和原始谱面保留在忽略的 `runs/` / `out/` 下，不进入 Git。每个分片的摘要仅覆盖 JSON 成员名与内容，**不是完整音频分片哈希**。

## 标签与划分口径

准备了 jumps、bursts、streams、stamina、slider_tech、reading 六个候选字段；它们只用于复核表结构，不代表这些类别已经具备可训练的数据。每张图的值初始均为 `null`，标签来源和复核者同样为空；未知不转成负标签。

`tags` 是原始自由文本。程序只统计独立词项与 `skillset/` 前缀作为检索线索，不从曲名、流派、关键词或密度直接生成技能真值，不把曲包标签当成各难度标签。实际训练分类仍需按覆盖、分歧和人工复核结果确定。

旧节奏训练按 `beatmapset_id` 哈希划分 3% 验证集；本次使用相同公式及至少 50 物件的过滤，检查同一预处理音频或来源音频哈希是否跨训练与验证。来源音频哈希仍是声明；没有解码核对同曲不同编码、剪辑和近重复，也没有据此冻结独立测试集。

## 启动训练前仍需完成

1. 确认可用的逐难度技能标签来源、版本、使用条件，完成抽样人工复核与一致性记录。
2. 核对音频身份和近重复，排除此前开发、验收用歌；按音频组冻结训练、开发、测试清单。v0 已训练歌曲不能被宣称为 v0 微调的全新独立测试歌。
3. 给节奏与坐标模型实现标签条件、缺失掩码、条件 dropout 和无标签路径，固定小规模微调与消融配置。
4. 通过 M1.1 后再启动 M1.2；遵守现有训练、独立验收和候选发布顺序。

## GPU 与验证

2026-10-08 14:41（北京时间），物理 GPU 0、1 在原项目环境中通过真实 CUDA 矩阵计算与反向传播，梯度有限；测试随后退出，显存释放。GPU 2、3 当时仍有其他账号的计算进程，本次没有使用它们。审计在 CPU 上执行；CUDA 检查不代表训练已经开始。

新增审计测试覆盖来源连接缺失、重复记录、缺失标签不被转成负样本、关键词边界、音频跨集合检测及已有人审目录保护。完整测试结果：**180 passed**。这不替代模型训练、真实生成、独立解析或客户端试玩。

## English status

M1.1 is in progress. All 74 shards were scanned and all 139,582 prepared maps were joined to source metadata. Only 1,815 maps contain literal skill-related terms, which remain unreviewed candidates; all skill labels remain null. A deterministic queue of 95 maps is ready for human review. The actual legacy loader independently confirmed 182 audio tracks across training and validation, affecting 721 of 4,220 validation maps. No label-conditioned training or new weights have been produced. GPUs 0 and 1 passed actual CUDA forward/backward checks; GPUs 2 and 3 were occupied by another account. Training still requires reviewed labels, an audited audio-grouped split and conditioned model inputs. The audit tests and full suite passed (180 tests).
