# 初始训练配置失败记录

日期：2026-11-13。

`pilot_dev_aug_s1` 使用全网 Adam lr=0.0015、没有分类器固定骨干预热、8×80 步短周期。CE 最后为 6.4773（651 类随机分类约为 ln(651)=6.4785），干净开发验证集 mAP=3.9099%、Rank-1=3%。模型未形成有效身份判别能力。

随后的 `pilot_dev_visibility_s1` 在训练期间也呈现接近随机的身份损失，因此主动停止整个系列。这组数据只能说明训练配置失败，不能用来支持任何创新有效性；已完成的 aug 结果及原始日志保留。未完成的 visibility 不进入有效消融表。

官方配置 `code/deep-person-reid/configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml` 使用 classifier-only fixbase 10 epochs、AMSGrad、150 epochs。当前低预算研究配方改为较小骨干 lr=0.0003，classifier/reliability lr 倍数5，2轮固定骨干预热、AMSGrad。该修正仍不等价于完整复现官方论文，需要重新检查其身份损失和验证表现。所有变体统一使用修正后的配方。
