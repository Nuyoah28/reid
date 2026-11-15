# 第四阶段 MDTA 结构筛选

## 目的与实现

v3 的普通空间 Transformer 两 seed 都低于 baseline。为测试不同的注意力维度，在 OSNet `[B,512,16,8]` 特征图后加入一个 MDTA-only 通道注意力残差 adapter：逐空间位置 channel LayerNorm，`1×1 QKV`、depthwise `3×3`，把 Q/K 沿 128 个空间点归一化并计算每头 `C/head × C/head` 通道关系矩阵，再混合 V、投影回 512 通道，以初始 0.05 的残差尺度加入原特征。没有加入 Restormer 的 GDFN。

实现兼容 `mmi-unet` 的 PyTorch 2.5/CUDA 12.1。它引用并改写自图像复原任务的注意力构件，**不是原创模块**。Restormer 来源为 Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022, DOI `10.1109/CVPR52688.2022.01167`。本地文献清单中的 *Occlusion-Aware Transformer With Second-Order Attention for Person Re-Identification*，TIP 2024, DOI `10.1109/TIP.2024.3393360`，标注为未取得全文；与它的精确机制重合尚未查明，因此不作排除重合的声明。

训练严格沿用 v3 的 8×80 PK、651/100 Market 身份划分、ImageNet OSNet 初始化、单视图 CE + 0.5 Triplet 和 Adam AMSGrad；每个 seed 的 MDTA 结果与冻结的同 seed v3 baseline 配对。两次 target evaluation 使用同一个预注册 query/gallery 清单 SHA256：`0c5ca34bc0cd6a68a8bed65b4e8f9e5cb4d38433cfa7280310f7d38e5f0d1104`。Occluded-REID 已被先前阶段查看，所以只作为探索性迁移结果。

## 结果

| seed | 开发遮挡 mAP | baseline 开发 mAP | 配对开发差值 | 真实 mAP | baseline 真实 mAP | 配对真实 mAP 差值 | Rank-1 | baseline Rank-1 | 参数量 | 更新 / AMP 跳步 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.76 | 11.89 | +1.87 | 40.59 | 40.14 | +0.44 | 47.60 | 47.20 | 3,232,937 | 638 / 2 |
| 2 | 11.69 | 11.98 | −0.30 | 38.50 | 39.13 | −0.63 | 45.30 | 46.20 | 3,232,937 | 637 / 3 |

| 指标 | MDTA 两 seed 均值 ± 样本标准差 | 同 seed baseline 配对差值均值 ± 样本标准差 |
|---|---:|---:|
| 合成开发遮挡 mAP | 12.72 ± 1.47 | +0.79 ± 1.53 |
| Occluded-REID mAP | 39.54 ± 1.48 | −0.09 ± 0.76 |
| Occluded-REID Rank-1 | 46.45 ± 1.63 | −0.25 ± 0.92 |

参数量为推理模型（排除训练分类器）：baseline 2,169,508，MDTA 3,232,937，新增 1,063,429，约增加 49.0%。4060 smoke test 使用 24 张输入，loss 有限、输出为 `[24,512]`，峰值 CUDA 分配约 1,058 MiB。运行 verifier 核对了 baseline checkpoint、协议和源码哈希、训练日志、有效更新数、评测条目和两次 target hash；核验通过。MDTA 专项三项单元测试通过。

## 研究决策

开发集 seed 1 上升、seed 2 下降；真实集 mAP 与 Rank-1 的平均配对差值略为负值，且 seed 方向相反。该结构没有显示稳定增益，**不保留为当前论文方法候选，只作为已完成的负向结构对照归档**。两个 seed 和八轮训练不是显著性或收敛证据，Occluded-REID 的结果也不是盲测。

Restormer MDTA 移植本身不能作为原创贡献；更重要的是，缺少 TIP 2024 二阶注意力论文全文，无法排除结构层面的相似工作。当前仅值得继续评估 v3 的 Semantic Detail Fusion，但它相对 baseline 的平均真实 mAP 优势也只有 +0.70 个百分点且 seed 方向不一致；需先核实相关工作并在未用于选择的数据集和更充分预算下复验，不能直接写成已证实创新。

逐项数据、差值和审计文件见[完整汇总](../results/occlusion_v4_summary/table.md)、[机器可读指标](../results/occlusion_v4_summary/metrics.json)、[档案核验](../results/occlusion_v4_summary/verification.json)与[预注册协议](occlusion_v4_protocol.json)。
