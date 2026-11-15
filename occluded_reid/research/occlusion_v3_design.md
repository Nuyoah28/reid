# 第三阶段结构候选：全局语义筛选高频细节

本阶段尝试在官方 OSNet x1.0 末端增加单层 Transformer 型空间上下文模块，并测试一个自设的语义筛选高频细节融合候选。研究问题是：全局人体表征能否给 16×8 的局部细节位置排序，并将更可能属于身份线索的高频响应注入空间特征？这只是可证伪的结构候选；高频也可能主要来自遮挡物边缘或背景。

## 论文来源与边界

| 来源 | 相关内容 | 本项目处理 |
|---|---|---|
| Zhou et al., **OSNet**, ICCV 2019，官方 MIT 实现 | OSNet x1.0、OSBlock、ChannelGate、FC | 骨干及 FC 沿用，不属于本轮创新 |
| Li, Chen, Ye, **Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification**, ECCV 2024, LNCS 15067, DOI `10.1007/978-3-031-72784-9_17` | 野生动物 ReID。ODS 用 ViT CLS-to-patch 注意力给高频 token 排序；FMA 在 FFT 频域混合；Feature Equilibrium 用 Huber 约束高频/原始特征 | 借鉴“全局语义评分、筛选局部细节”的研究思路。没有复制 ViT、FFT 区域混合、Huber equilibrium 或原模型；OSNet 的 pooled low-frequency feature 作为全局 query，局部 3×3 high-pass feature 作为 value，并加入本项目设计的空间残差门控 |
| He et al., **ReIDMamba: Learning Discriminative Features With Visual State Space Model for Person Re-Identification**, IEEE TMM 2026, DOI `10.1109/TMM.2026.3668499` | 双向 Mamba、多粒度 class-token 分支融合、ranking-aware triplet | 这是 Mamba ReID 的直接先例。没有移植其骨干、MGFE 或 RATR；本轮改用环境自带 PyTorch Transformer 模块，以隔离局部结构的作用 |
| Huang, Yoon, **SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-Identification**, CVPR 2026 | 依赖 SS2D 状态更新的 token pruning/merging，主要针对效率 | 视为 Mamba/ReID 先例；压缩和剪枝不是本阶段研究目标，也未复制其模块 |
| Wang et al., **Feature Completion Transformer for Occluded Person Re-identification**, IEEE TMM 2024, DOI `10.1109/TMM.2024.3379908` | 双流完整/遮挡配对、可学习 completion tokens、特征补全 decoder、FC²/CHT | 遮挡 Transformer 补全已有先例。没有移植 completion decoder 或双流损失，避免把已有的缺失特征补全概念改名为新贡献 |

ECCV 2024 野生动物 ReID 的高频消融提醒：只加高频增强可能受背景高频噪声影响而下降。因而本阶段保留 GAP 与普通 Transformer 对照，并完整记录负结果。论文库 PDF、论文清单和全文索引在共享 `reference-paper/` 与本地 `research/paper_text/`；全文提取目录不进入 Git。

## 结构定义

OSNet 输入为 `[B,3,256,128]`，末端输出为 `F∈R^(B×512×16×8)`，含 128 个空间位置。检索仍是一个全局 512 维 embedding。

### Baseline

```text
image -> official OSNet x1.0 feature maps [512,16,8]
      -> global average pooling [512]
      -> FC(512,512) -> BN -> ReLU -> L2 normalize
```

### 普通 Transformer 对照

把 `F` 展平为 128 个 512 维 token，加可学习 16×8 位置向量，经一个 pre-LayerNorm 8-head self-attention block 和 ratio-2 MLP，再还原到 16×8 后 GAP。Attention/MLP 残差 LayerScale 初始化为 0.1。这是用于比较的通用 Transformer encoder，不主张为本项目原创。

### 自设候选：Semantic Detail Fusion

先以 `3×3` 局部平均获得低频响应 `L`，定义高频残差 `H=F−L`。全局 query 来自 `GAP(L)`；位置 key 也来自 `L`，value 来自 `H`。对 query/key 做 L2 归一化，计算每个位置的 cosine score，只在每张图 score 最高的 64/128 个位置上归一化注意力：

```text
q = normalize(Wq GAP(L))
k_i = normalize(Wk L_i)
v_i = Wv H_i
a_i = softmax(q·k_i / 0.1), only over top-50% positions
c = Wo Σ_i(a_i v_i)
g_i = sigmoid(Wg [F_i, c])
F'_i = F_i + α g_i c,  α initialized to 0.1
```

这不是 AHT 的原模块复现：没有 ViT CLS、FFT/FMA、高频特征均衡损失；改为 OSNet 全局特征 query、局部平均高通、top-50% 候选和 identity-loss-trained spatial residual gate。也不声称“全局 token 选择局部细节”这一宽泛概念新颖。可被检验的新组合是将这种语义筛选应用到 OSNet 局部高频特征，并通过空间门控进行残差融合；它的有效性需由成组实验决定。

## 冻结比较协议

三变体为 `baseline`、`transformer`、`semantic_detail`；每个 seed 使用完全相同的 Market 训练身份划分、采样器和 ImageNet 初始化。使用 8 epoch × 80 PK batches、每批 6 身份 × 4 张源图；单视图 label-smoothed CE + 0.5 batch-hard triplet。前两轮冻结 OSNet feature extractor，Adapter/FC/分类器使用 5×学习率。不同于 v2 三视图对照，本阶段三种结构都训练同一个普通单视图配方，避免把复杂监督同时叠加进结构比较。

开发指标是 30%/50% 横向条带合成遮挡 mAP 均值。全部六组完成后才冻结 checkpoint SHA，再统一评测 Occluded-REID 的 1000 query × 1000 gallery。前两阶段已经查看过该目标域，因此这里只称探索性迁移。完整注册协议见 [occlusion_v3_protocol.json](occlusion_v3_protocol.json)，显存 smoke test 见 [occlusion_v3_smoke.json](occlusion_v3_smoke.json)，运行后结果见 `../results/occlusion_v3_summary/table.md`。

训练环境仍使用已授权的 `mmi-unet`、PyTorch 2.5.0、CUDA 12.1 和 RTX 4060 Laptop 8GB。新代码在 `code/occlusion_v3/` 与 v3 独立入口；不改写 v1/v2 模型和冻结档案。短预算、两 seeds、CUDA 非确定性及 Occluded-REID 既往暴露都限制结论强度。
