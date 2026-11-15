# 第四阶段 MDTA 结构筛选

百分制；配对差值以相同 seed 的 v3 baseline 为参照。两 seed 的样本标准差不是置信区间。

| seed | 开发遮挡 mAP | 真实 mAP | 真实 Rank-1 | 开发差值 | 真实 mAP 差值 | 参数量 | 更新 / 跳步 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.76 | 40.59 | 47.60 | +1.87 | +0.44 | 3,232,937 | 638 / 2 |
| 2 | 11.69 | 38.50 | 45.30 | -0.30 | -0.63 | 3,232,937 | 637 / 3 |

## 两 seed 均值 ± 样本标准差

| 指标 | MDTA | 配对差值 |
|---|---:|---:|
| Synthetic occlusion development mAP | 12.72 ± 1.47 | +0.79 ± 1.53 |
| Occluded-REID exploratory mAP | 39.54 ± 1.48 | -0.09 ± 0.76 |
| Occluded-REID exploratory Rank-1 | 46.45 ± 1.63 | -0.25 ± 0.92 |

真实目标数据清单在训练前注册，checkpoint 在全部开发训练结束后统一冻结。Occluded-REID 已在前序阶段暴露，因此仅作探索性迁移。

完整逐项结果与 hash 见 `metrics.json`、`external_freeze_manifest.json` 和 `verification.json`。