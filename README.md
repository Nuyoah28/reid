# ReID 研究工作区

当前主线确定为 **遮挡行人重识别**。三个方向独立保存，参考论文统一保存；不把轻量化作为当前论文的目标或贡献。

| 目录 | 定位 | 入口 |
|---|---|---|
| `occluded_reid/` | **当前主线**：遮挡身份线索、可靠性校准与真实遮挡泛化；含 OSNet baseline、历史实验和新模块对照 | [项目说明](occluded_reid/README.md) |
| `infrared_multimodal/` | 已有红外多模态原型；本轮不扩展此方向 | [已有原型](infrared_multimodal/README.md) |
| `lightweight_occluded_reid/` | 用户确认保留的历史方向归档；不继续做轻量化研究 | [归档边界](lightweight_occluded_reid/README.md) |
| `reference-paper/` | 用户提供的统一论文清单与 PDF，所有方向共用，本地保留 | 不上传 PDF |

训练请先进入对应方向：

```powershell
conda activate mmi-unet
cd E:\论文\occluded_reid
python -m unittest discover -s code/tests -v
```

2026-11-14 整理只迁移原遮挡项目目录；旧实验的配置、日志和源码原文保留。历史记录中 `E:\论文\code`、`data`、`runs`、`research`、`results` 等路径现在对应 `E:\论文\occluded_reid\` 下同名目录，不能把目录迁移误认为新的实验。红外原型已有目录未重建。共享论文仍在工作区根目录。

根 `.gitignore` 统一排除各方向的数据、权重、checkpoint、运行缓存和基准照片，保留可审计的代码、配置、文本日志、结果 JSON/CSV 和研究说明。大文件与参考论文不进入 Git。
