# 遮挡行人重识别：跨图身份证据与可靠性校准

当前唯一主线为**遮挡 ReID**。轻量化只保留历史归档，不作为研究目标或贡献。

## 第五阶段：交叉因子反事实遮挡（待实验）

前序结果没有证实可靠性、剩余证据一致性或结构适配器对真实遮挡有稳定优势。当前注册的可证伪假设是：固定矩形遮挡几何，对两个目标身份复用相同的两种跨图 donor patch，显式压低“身份 × donor patch 内容”的差分交互项。patch 未分割，包含人体或背景内容，不等同纯遮挡物。该 2×2 目标是项目待检验设计；COPE、跨图 patch 增强与 counterfactual ReID 都是邻近先例，现有检索未发现完全同式实现，但不能称作原创贡献或有效方法。

- [v5 研究设计与失败判据](research/occlusion_v5_design.md)
- [v5 固定协议、种子与比较组](research/occlusion_v5_protocol.json)
- [首次启动失败记录（零 optimizer update）](research/occlusion_v5_startup_incident.md)

矩阵含 4 个对照、5 个配对 seed、每组 2,000 optimizer steps。开发集用 Market 留出身份和固定 donor patch 遮挡；Occluded-REID 在全部模型完成训练并冻结 source/data/checkpoint hashes 后统一评测，仅作探索性迁移证据。正式启动：

```powershell
python tools/run_occlusion_v5.py --print-plan
python tools/run_occlusion_v5.py
python tools/verify_occlusion_v5.py --require-checkpoints
python tools/summarize_occlusion_v5.py
```

两次启动故障已留档：首个 tag 因协议权重字段缺失而零更新退出；`v5b` 的 s11 完成训练但因开发评测漏导入指标函数而未产出结果。正式矩阵使用新 tag `occlusion_v5c`，重新运行所有组以保证同一源码版本，失败配置、history、日志和 checkpoint 均保留。

第二阶段 v2 新增“排除自身源图的身份原型 margin＋双干预最小支持”监督候选，沿用 OSNet 骨干与 v1 可靠性头。六变体、两种子的有限预算实验已全部完成；全部开发训练完成后冻结 checkpoint 清单，再统一外测全部变体。

**实测结论：本轮新候选没有真实遮挡优势，暂不能作为最终论文方案。** 普通 OSNet 对照的真实 Occluded-REID mAP / Rank-1 为 **43.94% / 51.50%**（两种子均值）。`peer`、`peer_stability`、`peer_evidence` 的真实 mAP 分别为 **37.35%、36.45%、36.34%**；联合目标虽提高合成开发指标，却未转化为真实提升。完整负结果、每个 seed 和配对比较均已保留。

## 第三阶段结构对照

v3 固定同一 OSNet x1.0 骨干、单视图 CE+Triplet 配方与训练身份划分，对照 `baseline`、一层 8-head 空间 Transformer，以及借鉴 ECCV 2024 高频筛选思路并加入自设空间残差门控的 `semantic_detail`。**普通 Transformer 两种子都低于 baseline；semantic_detail 的两种子真实 mAP 均值为 40.34%，较 baseline 的 39.64% 高 0.70 个百分点，但配对方向随 seed 改变，且 AHT/DPEFormer 已有相近的全局引导局部筛选机制。** 该支线仅保留作机制分析，不再作为清晰的论文原创候选。

- [v3 实验报告：逐 seed 结果、配对差值和研究决策](research/第三阶段结构实验报告.md)
- [v3 完整网络图：baseline、OSBlock 卷积／采样细节、两个结构分支与论文来源](research/architecture/occlusion_v3_network.html)
- [冻结设计与来源边界](research/occlusion_v3_design.md)、[预注册协议](research/occlusion_v3_protocol.json)
- [六组原始结果、对照图和配对汇总](results/occlusion_v3_summary/table.md)、[档案核验](results/occlusion_v3_summary/verification.json)

## 第四阶段 MDTA 对照

v4 只把 Restormer 启发的单层 MDTA 通道注意力用于结构对照；相对同 seed v3 baseline，真实 mAP 两 seed 平均差值为 **−0.09 个百分点**，Rank-1 平均差值为 **−0.25**，两 seed 方向相反。**本轮不保留 MDTA 为方法候选，也不把其论文模块移植算作原创贡献。** 相关 TIP 2024 二阶注意力论文全文尚未取得，具体机制重合仍不确定。

- [v4 MDTA 实验报告与来源边界](research/第四阶段MDTA实验报告.md)
- [v4 两 seed 配对结果](results/occlusion_v4_summary/table.md)、[档案核验](results/occlusion_v4_summary/verification.json)
- [v4 预注册协议](research/occlusion_v4_protocol.json)、[RTX 4060 smoke test](research/occlusion_v4_smoke.json)

- [第二阶段实验报告：结果、负结果和研究决策](research/第二阶段实验报告.md)
- [方法与来源：公式、梯度及逐项原创边界](research/第二阶段方法与来源.md)
- [完整网络图：baseline、各卷积／采样块及新训练分支](research/architecture/第二阶段网络图.html)
- [全部 12 组结果与配对差值](results/peer_v2_summary/table.md)
- [DPM、RGANet 等相关工作的查重](research/原型方法查重记录.md)
- [预注册协议](research/occlusion_v2_protocol.json) / [档案核验](results/peer_v2_summary/verification.json)

## Baseline 与新候选

Baseline 为 Zhou 等的 **OSNet x1.0，ICCV 2019**。[官方 deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) 已经 Git 克隆，固定版本 `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` 的源码和 MIT LICENSE 保存于 `code/deep-person-reid/`。本项目继承官方骨干，使用 ImageNet 初始化；自写 Windows 训练引擎不等同于复现原论文完整配方或指标。

新 target 只在训练期构造：用同身份其他源图形成正原型，与批内其他身份原型作局部余弦 margin 比较；排除自身及重复副本，重复源图总权重为一票。取两个 donor 遮挡视图的较低 margin，形成停止梯度的空间监督。推理仍为一张图进入 OSNet 和可靠性头，输出一个 512 维检索向量。

| 变体 | 输入 | 池化／监督 | 用途 |
|---|---|---|---|
| repeat3 | 同一已增强 tensor 复制三份 | GAP | 匹配图像数预算；非三次独立增强 |
| paste3 | clean＋两个矩形 donor | GAP | 普通遮挡增强对照 |
| stability | 同上 | v1 头，G×A | 旧外观稳定性监督 |
| peer | 同上 | 相同头，G×I | 新跨图身份支持监督 |
| peer_stability | 同上 | 相同头，G×I×A | 新监督与旧稳定性联合 |
| peer_evidence | 同上 | G×I×A＋v1 一致性损失 | 检验直接复用旧损失的增量 |

G 是几何项，A 是外观稳定性，I 是身份支持；都不是人体可见性真值。原型学习、判别性与不变性联合评估已有论文先例，本阶段新增的是具体监督构造，不能声称首次提出这些宽泛概念。

## 环境与数据

使用用户确认的现有 **mmi-unet**：Python 3.10.15、PyTorch 2.5.0、torchvision 0.20.0、CUDA 12.1、RTX 4060 Laptop 8GB。本轮未替换训练环境包；快照见 [环境](research/environment.json) 和 [包清单](research/environment-pip-freeze.txt)。

所有命令在**本方向目录**执行：

```powershell
conda activate mmi-unet
cd E:\论文\occluded_reid
python -m unittest discover -s code/tests -v
```

本机解释器为 `D:\pytorch\Anaconda3\envs\mmi-unet\python.exe`。训练入口直接加载官方 osnet.py，无需安装整个上游 requirements。PDF 工具在独立文档 Python 中运行。

本机已准备数据和预训练。新机器准备命令及布局：

```powershell
python tools/download_assets.py imagenet
python tools/download_assets.py market
python tools/download_occluded.py
```

```text
data/Market-1501-v15.09.15/{bounding_box_train,query,bounding_box_test}/
data/occluded_body_images/<id>/*.tif
data/whole_body_images/<id>/*.tif
weights/osnet_x1_0_imagenet.pth
```

来源和 SHA256 见 `research/asset_*.json`、`research/primary_source_checks.json`。只用 ImageNet 预训练，不采用已经在 Market 或混合 ReID 数据训练的权重。数据、照片与大权重不上 Git；共享论文在上一级 reference-paper/。

## 第二阶段复跑与核验

固定 Market 训练身份内部划分：651 个训练身份、100 个留出身份；每组 8 轮×80 个 PK 批次，每批 6 身份×4 次采样，三视图共 72 张输入。首两轮冻结骨干，评测固定最后一轮。AMP 跳步单独记录，640 个批次不一定等于 640 次实际参数更新。

```powershell
# 新 tag 保留已发布档案；顺序完成全部 12 组，再统一外测
python tools/run_occlusion_v2.py --tag replica_v2
python tools/verify_occlusion_v2.py --tag replica_v2 --require-checkpoints
python tools/summarize_occlusion_v2.py --tag replica_v2

# 核验本轮已归档的 peer_v2
python tools/verify_occlusion_v2.py --require-checkpoints
python tools/summarize_occlusion_v2.py
```

同一 tag 可以恢复本地未完成 checkpoint；已有档案但缺少本地 run 时必须另用 tag，避免覆盖。恢复时源码、划分和参数必须保持不变。新机器只核验小型档案时，省略 --require-checkpoints；这不等于核实未取得的大权重。根 .gitattributes 固定本轮审计涉及的源码与结果的 LF/CRLF，保留可复核的字节哈希。

开发主指标为 30% 与 50% 合成噪声 mAP 的均值；它不是现实遮挡数据集成绩。真实评测为 Occluded-REID 的 1000 张遮挡 query 对 1000 张完整 gallery、200 个身份，无 reranking、无翻转集成。v1 已观察该真实数据域，v2 只能称**探索性跨域确认**。两种子短训练不支持显著性、充分收敛或 SOTA 声明，不能与不同训练身份和预算的旧结果直接计算模块增益。

本轮 12 组共 7,680 个 PK 批、7,671 次实际更新，9 次 AMP 跳步；逐轮训练时间合计约 44.0 分钟，不含加载、评测和失败启动。期间一次训练启动、一次外测进程在 `nvcuda64.dll` 崩溃；保存故障记录后，使用原配方／冻结权重完成恢复。根因尚未查明，详见 [报告中的协议、资源与故障记录](research/第二阶段实验报告.md#2-冻结协议与实际预算)。

## 历史 v1 与审计

v1 的 plain / aug / visibility / full 在 751 身份、20×100 批次、seed=1 下，真实 Occluded-REID mAP 为 **55.73 / 48.50 / 44.41 / 46.85**。旧完整方案未获得真实遮挡优势；这些负结果与原始源码保留。

- [v1 实验记录](research/实验记录.md)、[旧完整结果](results/official_summary/table.md)、[旧网络图](research/architecture/网络结构图谱.html)
- [v1 研究方案](research/研究方案.md)、[旧创新来源](research/创新来源清单.md)、[历史下一步决策](research/下一步研究决策.md)
- [21 项实现测试日志](results/unit_tests_v2.log)、[v1 迁移后核验](research/v1_post_migration_verification.json)
- [230 个旧代码／结果文件迁移审计](research/workspace_migration_audit.json)
- [近期遮挡论文指标与协议对照](research/近一年遮挡ReID成果对照.md)、[BibTeX](research/references.bib)

历史日志中的 E:\论文\code、runs、data 等旧路径，现在对应 E:\论文\occluded_reid\ 下同名目录；未为迁移而改写原始记录。投稿仍需有效机制、充分训练、现代遮挡方法公平对照及独立确认数据集。
