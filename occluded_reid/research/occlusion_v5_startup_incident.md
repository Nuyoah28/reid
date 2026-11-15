# v5 启动检查记录

2026-11-15 首次 runner 冒烟调用未进入训练：训练在创建首个 optimizer step 前因协议 JSON 未包含 `pairwise_weight` / `interaction_weight` 而抛出 `KeyError`。该启动留下的 `runs/occlusion_v5_plain_s11/config.json` 和 `results/occlusion_v5_plain_s11/train.log` 保留作审计，未生成 checkpoint、history 或结果，也没有完成 optimizer update。

正式协议现已显式登记两个权重，并使用 run tag `occlusion_v5b` 开始矩阵，避免覆盖该失败记录。该启动失败不属于正式实验样本，也不计入 registered runs。
