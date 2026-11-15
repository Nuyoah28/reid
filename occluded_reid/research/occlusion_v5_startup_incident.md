# v5 启动检查记录

2026-11-15 首次 runner 冒烟调用未进入训练：训练在创建首个 optimizer step 前因协议 JSON 未包含 `pairwise_weight` / `interaction_weight` 而抛出 `KeyError`。该启动留下的 `runs/occlusion_v5_plain_s11/config.json` 和 `results/occlusion_v5_plain_s11/train.log` 保留作审计，未生成 checkpoint、history 或结果，也没有完成 optimizer update。

随后 `occlusion_v5b_plain_s11` 完成 20 轮、2,000 次 optimizer updates 并写出 checkpoint/history，但最终开发评测因训练入口漏导入 `evaluate_rank` 而失败，未生成结果文件。其 config、逐轮 history、完整 train.log 与本机 checkpoint 均保留；该 run 不属于正式注册矩阵，结果不用于研究判断。

修正导入后，协议使用新的 run tag `occlusion_v5c`，以统一源码重新运行完整矩阵。两次失败都保留作审计，不覆盖、不混入正式结果。
