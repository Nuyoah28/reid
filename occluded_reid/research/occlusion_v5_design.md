# v5 研究设计：身份与 donor patch 内容交叉效应

## 问题与假设

前序 v1/v2 的可靠性和剩余证据一致性没有稳定改善真实 Occluded-REID；v3 的 SemanticDetailFusion 两个种子方向相反，且全局语义筛选局部高频特征已有 AHT、DPEFormer 等相关先例。因此本轮不把这些模块继续包装为候选原创贡献。

本轮改为检验一个更窄的问题：固定遮挡几何时，换跨图 donor patch 内容造成的特征变化是否依赖目标身份？如果两种身份对同一 donor patch 变化产生不同反应，这个 identity-by-donor interaction 会让训练中合成的干扰外观与身份特征纠缠。交叉效应损失试图压低这种依赖，同时容许两种身份都出现的共同变化。当前 patch 没有做人/遮挡物分割，包含人体或背景内容，不能称为纯 occluder appearance intervention。

对同一遮挡框构造目标身份 `A/B` 与 donor 外观 `d1/d2` 的四视图：

```text
z_A1 = f(A with donor d1)    z_A2 = f(A with donor d2)
z_B1 = f(B with donor d1)    z_B2 = f(B with donor d2)

L_interaction = ||(z_A1 - z_A2) - (z_B1 - z_B2)||_2^2
```

这只约束差分中的交互项，不要求 `z_A1 = z_A2`，也不要求身份主效应或 donor 主效应为零。交叉项损失在同身份 pairwise consistency 之上提供直接对照，检验较弱的“身份条件化干扰”约束能否比完全不变性更合适。

## 方法边界

该 2×2 设计借用一般因子实验/反事实比较的逻辑，尚未通过穷尽文献检索证明新颖。不得声称首次提出因子分解、反事实 ReID、差分中的差分或遮挡不变性。训练 donor 仍来自当前 batch 的 Market-1501 原图像 patch；这是受控合成干预，不等同真实人体遮挡，也不能保证 donor patch 只包含遮挡物。

相关遮挡 ReID 工作已有直接邻接：COPE（CVPR 2026）做跨身份共享遮挡一致性和 CLIP 引导前景处理；FLaN-Net（IJCAI 2025）使用主体、可见属性和遮挡物 prompt，并按不确定性融合；AHT（ECCV 2024）用全局语义注意力选择高频 token；DPEFormer（KBS 2025）用 CLS proxy 做动态 patch 筛选并聚合局部部件；FCT（TMM 2024）用完整/遮挡双流和 completion decoder。SPT、DPEFormer 的 mask copy-paste 和 VersReID 的 MPDA 也使用跨图 donor patch 增强；CAL（ICCV 2021）已在 ReID 中研究 counterfactual attention。已查本地方法页未发现完全相同的「同一 donor crop × 两个目标身份 × 两种 donor 内容 + 全局 embedding difference-in-differences」公式，但检索不穷尽且与上述机制邻近，不能声称首创。v5 不复用这些论文模块，需继续核验更广泛的 identity-by-donor factorial prior art。

## 预注册对照

协议文件为 [occlusion_v5_protocol.json](occlusion_v5_protocol.json)。使用 OSNet x1.0、固定 Market 身份划分和 ImageNet 初始化，五个配对种子，每种子 20 轮 × 100 步。四组分别为干净图 plain、双遮挡外观普通增强、同身份双外观 cosine consistency、2×2 interaction loss。三个遮挡训练组每步处理 48 个合成视图；plain 每步处理 24 张原图，作为架构能力参照，不与其声称等计算量。数据清单包含相对路径、身份、摄像头和每张图的内容 SHA256。

开发集身份与训练身份不重叠。每个开发身份以固定规则选择一个 query camera，该摄像头所有图像作为 query，其余摄像头作 gallery，以增加 query 数并避免 query/gallery 同摄像头正样本。query 的遮挡是固定矩形 donor patch，donor 只从 Market 训练身份取样，几何与 donor 对所有模型相同；30%/50% donor 区域 mAP 均值是主指标，clean mAP、Rank-1、Rank-5 和 mINP 为辅助指标。保留全部种子，报告逐 seed 配对差和均值±样本标准差；五种子不是显著性检验。

Occluded-REID 已在前序阶段看过，只在全部开发训练完成、检查点冻结后对所有模型做一次跨数据集评测。它仅作探索性迁移结果，不参与模型选择。若后续要确认泛化，需要另一个此前未查看、且明确获授权的数据集。

## 失败判据

若 interaction 组在固定开发主指标上不能稳定超过普通增强及 pairwise control，或其提升只来自干净/单档合成条件，则不保留为正向方法主张。即使开发集表现为正，也必须在独立遮挡集和强基线对照后才能讨论投稿贡献；Occluded-REID 的既有暴露结果不能充当盲测确认。
