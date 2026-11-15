# 近一年遮挡行人重识别成果对照

检索与核查日期：2026-11-13；时间范围：2025-11-13 至 2026-11-13。本记录为对已查到论文原文的选择性核查，并非穷尽检索；不能据此断言全领域最高成绩。表中数值是作者报告，未独立复现。

## 评测指标和协议

Rank-1 是查询图片的首个检索结果身份正确的比例；mAP 衡量所有正确匹配的检索排序质量。二者均为百分比，比较时不能只看 Rank-1。

Occluded-Duke 通常使用该数据集的训练集训练和测试。Occluded-REID 没有单独训练集；下面三篇作者均采用 Market-1501 训练、Occluded-REID 测试的跨数据集协议。即便名称相同，预训练、骨干、遮挡增强、检索方法、训练预算和评测实现仍可能不同，因此各论文数字仅供了解量级。

## 近期代表性结果

| 方法和发表状态 | 视觉骨干和主要特点 | Occluded-Duke Rank-1 / mAP | Market→Occluded-REID Rank-1 / mAP | Market-1501 Rank-1 / mAP |
|---|---|---:|---:|---:|
| [COPE，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_COPE_Consistent_Occlusion_and_Prompt_Enhancement_Network_for_Occluded_Person_CVPR_2026_paper.html)，Table 1 | CLIP ViT；遮挡增强、前景提示和利用库中近邻的 PSS 检索 | 82.1 / 75.4 | 88.7 / 88.2 | 96.2 / 93.0 |
| [FLaN-Net，IJCAI 2025](../../reference-paper/R0513_B_2025_IJCAI_Categorical_Attention_Fine_grained_Language_guided_Noise_Filtering_Network_for_Occluded_Person_Re_Identification.pdf)，Table 1 | CLIP ViT-B/16 + CLIP text；主体、可见属性、遮挡物 categorical prompts 与不确定性加权 | 75.2 / 65.5 | 92.6 / 89.5 | — |
| 同一论文 COPE* | 重叠滑窗设定，论文 Table 1 带星号结果 | 82.4 / 76.4 | 90.9 / 90.7 | 96.3 / 93.3 |
| [DPM++，2026-05-07 arXiv 预印本](https://arxiv.org/abs/2605.06637)，Table 1–2 | CLIP ViT-B/16；动态掩码度量及显著性引导的遮挡合成；作者使用四张 RTX 3090 | 76.9 / 67.2 | 95.4 / 91.8 | 96.1 / 91.2 |
| [DPL-ReID，2026-05-19 arXiv 预印本](https://arxiv.org/abs/2605.19527)，Table 2–3 | CLIP ViT-B/16 与 LSNet-T；双提示、遮挡增强和门控融合 | 76.3 / 67.2 | 94.9 / 91.3* | — |

\* DPL-ReID 论文 Table 2 和 Table 3 均给出 Occluded-REID Rank-1 为 94.9，但第 9 页正文写成 76.3；此处采用两个表中相同的数字，并保留该原文内部矛盾，未与作者核实。该预印本 Table 9 报告 DPL-ReID 的参数量为 142.12M，CLIP-ReID 为 126.68M；属于作者统计口径，不能在没有统一计数口径时严格比较。

COPE 的 82.1 / 75.4 含 PSS。其 Table 2 在 Occluded-Duke 上去掉 PSS、保留 CICO 和 PBF 时为 76.8 / 68.9；这提示库内近邻检索带来明显变化，不能当成完全相同的单向量余弦检索。论文 Table 7 另报重排序结果，不能混入上述 Table 1 数值。

这些论文展示了遮挡 ReID 的多种路线与较高成绩，不应因骨干或参数规模不同而直接排名。CLIP 主要作为预训练/训练辅助引入，本表所述任务的测试输入仍为可见光图像；不能因此称为红外—可见光跨模态 ReID。预印本尚需同行评审和复现验证。DPM++ 自称的最优结果是相对于其论文所列比较方法，不能推断其优于未列入比较的所有后续方法。

## 与本项目实验比较

本项目 [实验记录](实验记录.md) 中，Market→Occluded-REID：OSNet x1.0 plain 为 Rank-1 **62.10**、mAP **55.73**；当前 full 候选为 **52.00 / 46.85**。分别约 2.170M 和 2.186M 推理参数；使用 ImageNet 权重初始化、20 轮×100 更新、单种子、无 reranking。full 较 plain 分别低 10.10 和 8.88 个百分点。

这只是有限训练预算下的实验，plain 与 full 输入视图和计算量也不同；不能把差距简单归因于某一模块。论文模型的预训练、骨干、训练量与检索机制与本项目不同，上表差距不能作为公平的架构优劣结论。现有结果尚未证明候选方法对真实遮挡有效，不能声称已达到这些论文水平。

## 对选题和创新性的影响

- DPM++ 已提出按输入动态选择可靠身份子空间并开展遮挡一致匹配，与“根据遮挡选择可靠身份特征”的概念有明显重合。现有设想不能仅凭这个概念宣称首创，应逐项对比信息来源、训练监督、匹配方式和计算预算。
- COPE 已结合遮挡合成、前景定位及库内样本的信息；DPL-ReID 也研究真实遮挡增强和双提示。单独引入注意力、提示词或普通遮挡增强不构成充分创新证据。
- 可继续探索 **真实遮挡泛化与可见证据可靠性学习**，通过受控遮挡干预、机制诊断和跨数据集验证，回答合成遮挡监督为何会负迁移、如何改善真实遮挡检索。具体机制与新颖性需要更全面的文献检索和实验支持，不能保证达到 CCF C 录用要求。
- 下一步优先补充代表性现代遮挡 ReID 方法的公平对照，统一数据、预训练、训练预算、检索协议及是否使用重排序；报告 Rank-1、mAP 和误例分析。先分析目前合成遮挡增强对真实遮挡性能的负迁移，再决定是否重构方法。

另检索到 2026 年发表的 [DFLNet](https://doi.org/10.3390/s26165160) 和 [Composite Augmentation and Feature Integration Reconstruction](https://doi.org/10.3390/s26103114) 等相关论文，但本次未取得可核验的实验表，不在此列具体成绩或判断其优劣。
