# 离线源端优化几何：SAM 文献与实现核查（2026-10-04）

状态：文献/实现语义准备及一个合成恢复检查；没有新的UWB训练、没有冻结实验协议、没有效果或创新主张。主论文全文未读，不把作者README/代码当作全文评审。

## 可核查来源及阅读边界

Foret, Kleiner, Mobahi, Neyshabur, *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021，arXiv:2010.01412。作者库 https://github.com/google-research/sam 的README全文已读（blob9730c3031e34f6c6ab9be7b2b578b602a964323a）；包含摘要、CIFAR/ImageNet示例与JAX/Flax运行说明，不能外推UWB效果。README图像未视觉分析。作者 sam_jax/training_utils/flax_training.py（blob50be4bfdf4ddf8c21fdf6e25c213bb7dfd2d51cc，917行）仅读69–115、165–200、225–343、442–580、767–828；非完整源码通读。

另读非作者PyTorch实现 https://github.com/davda54/sam 的README全文（blob7499b90e8c14ab45af3709105c33c44867dac199）与sam.py全文（blobdf9bcb7d97791f2f3036c2b173b12f8c572d0d80）。它明确为非官方实现；不能称作者算法完整复现。没有把第三方代码复制入仓库。主文网络访问受当前package_managers网络预设限制，未绕过代理。

## 实现含义

作者实现以全参数全局L2梯度范数构造扰动，两次同批前向/反向，外梯度用于优化器更新。所读closure重复使用同一prng_key；随机层掩码、内外状态选择必须显式处理。其损失含耦合L2，并有排除bias/BN开关，不能等同当前AdamW解耦衰减。作者代码对BN保存内步状态；当前模型为GN但仍有Dropout(0.2)，因此没有BN运行统计不代表随机性问题消失。rho示例0.05只是文献超参数，不是UWB最优值或已冻结本实验参数。

所读第三方封装拥有独立base_optimizer。合成Linear(2,1)、固定3样本MSE、seed22、AdamW lr=.001/decay=.001、3步后，仅保存封装state_dict并恢复，base AdamW状态条目由2变0；再走一步，最大参数差3.921985626220703e-05。封装保存的参数状态只有old_p。见SOURCE_SHARPNESS_RESUME_CHECK.json和tools/check_sam_resume_review.py。结论限于这个blob的这种保存方式，需要显式保存/恢复嵌套base优化器。不是所有SAM库均有问题，更不是已有UWB实验发生恢复错误的证据。此前完成研究直接保存AdamW，其正确性另有独立审计。

## 与已有证据的关系及下一步

原项目py/md按 SAM(/ASAM/GSAM/sam_rho/sam_optimizer/sharpness-aware 等明确标记检索无对应实现命中；这是文本筛查，不是证明不存在别名实现，更不构成新颖性证据。既有SAR式在线GN无标签更新失败、SWAD平均失败；离线有标签全参数SAM的训练对象与部署协议不同，但不能由此预测有效。源训练错误率0而验证错误约7%的观察可提出优化泛化问题，不能证明sharpness是原因。

继续前需补核查作者主文与许可、已有源优化代码和可复现数值；如选择实现，明确两个梯度的目标/同批随机掩码、零范数/异常恢复、基础优化器状态、scheduler、尾平均在哪次真实更新后累积。用独立解析/有限差分小例、恢复连续等价与留出输入/标签隔离检查；先测真实CPU成本。实验须在任何候选诊断评分前冻结源训练/验证/诊断角色、rho及主选模规则、同模型同样本和计算预算对照，再完成全部六源先导。未满足这一步不启动完整八折。不能把SAM加GN/尾平均或多视角简单组合称创新；需要额外领域机制与有力对照支持。
