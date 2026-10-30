# ResNet：完整网络的原生 Keras CPU 准备

本轮为已发表强基线的可运行迁移准备，尚无真实 ResNet UWB 值、标签、fit 或准确率结果。固定 seed20261029。当前网络834124总参数（含BN非训练状态）、48可训练参数张量、98原生Adam变量；没有缩宽、缩层、缩批或删捷径。

文献依据：Wang, Yan & Oates, *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*, [arXiv:1611.06455](https://arxiv.org/abs/1611.06455)。本轮读摘要、II.B/C残差架构和III.A训练设置的文字/公式；未完整阅读9页或核验图像。其时间卷积残差块和GAP为当前方法依据，原文Adam epsilon1e-8与当前代码原生默认1e-7有差异。

Fawaz等，*Deep learning for time series classification: a review*, [arXiv:1809.04356](https://arxiv.org/abs/1809.04356), DOI10.1007/s10618-019-00619-1。本轮读摘要/引言、3.2.3 ResNet及实验设置/表1–2相关文字，不是完整44页/图审读。表2批量16与当前作者代码64不同；相关正文的通道描述也应以固定源码64/128/128为本配置依据。

作者基准库hfawaz/dl-4-tsc固定commit e0233efd886df8c6ca18e6f1b545d15aaf423627；5文件按Git blob SHA1和本地SHA256核验。保留完整GPL-3.0许可证。README区分论文TF1.14历史commit与当前TF2代码，因此不声称原论文精确复现。旧789快照有界检索未发现此完整Classifier_RESNET实现；历史通用ResNet文字并不是等价实验，检索也不构成新颖性证明。

直接从源文件AST提取build_model，唯一AST修改为best_model.hdf5→best_model.keras文件扩展名。调用真实Keras3.11.3/Torch CPU后端，无TensorFlow导入stub或手写网络替换。11 Conv1D、12 BN、3 Add、GAP、12类softmax完整保留，包括偏置、even-SAME非对称padding、第三块BN捷径。CE、原生Adam.001/epsilon1e-7/无weightdecay，ReduceLROnPlateau训练loss/factor.5/patience50/min_lr.0001，按最小训练loss保存best。固定12类来自既有任务分类定义；后续不得复制原main.py合并train+test标签拟合onehot/确定类别数的路径。

合成检查：独立NumPy卷积/BN移动矩/Adam方程；3完整B64更新与恢复后2步的权重、优化器、BN、指标及全部RNG逐位；原生.keras权重/优化器保存加载。首次快照因Keras global SeedGenerator惰性创建而记录不同时刻的Python RNG，保留两份失败日志及原脚本，已先物化global generator再快照，且现在强制立即恢复全状态一致，不跳过指标。

另用70合成记录、B64/末6，单次连续原生fit三轮与显式原生train_on_batch/test_on_batch循环比较，所有batch/epoch指标、权重/优化器/BN、原训练loss选中checkpoint均逐位相同。251个合成恒定loss回调符合独立LR规则。**未声称原生fit数据适配器的全局RNG相同**；真实入口仍需固定显式shuffle/实际ID日志、回调状态和独立重放，不能据准备检查跳过这些工作。

成本规则在测量前保存于COST_POLICY。一次独立600wall/590–600CPU进程测8完整train/4eval，无真实数据；maxtrain .436682109、maxeval .131066100秒。按2×最大步耗时+120估计，1500轮103363.775629秒>43200预算，预先定义的100轮7011.481360秒通过。因此仅固定**100轮的CPU开发配置**，保留完整网络和B64；它不是1500轮基准复现，未来100轮失败不能否定1500轮配置。成本进程exit0/no timeout，8.190444wall/8.140354childCPU；测量并非完成时长保证。

下一步具体工作：实现native Keras真实runner、逐步重放auditor、首步及私有备份守卫；冻结100轮/B64/seed20261029/source3–6训练、2验证、1诊断、90×567 RMS/原训练loss best和ReduceLR规则；每轮4700记录/74批末28、无earlystop，显式确定shuffle。协议/代码/配置远端预登记并逐字读回后才允许一次真实fit。两份最终预测冻结后才读held1标签，以相同reference .9866666666666667和+.005门槛关闭或另注册扩展；不得依据诊断选择seed/轮数/网络等，7–8标签不用。所有源结果探索性，既非旧八折也非独立测试。当前没有新后台训练。

依赖10个wheel锁定SHA，1302运行文件锁定（排除pyc/安装路径RECORD/direct_url）。Git保存requirements/哈希/恢复脚本及本次小代码和许可源码；不放wheel、.so、原数组/maps、实际样本ID或模型。restore_dependencies.py只恢复依赖，不是训练resume；Python/NumPy/Torch等基底另见RUNTIME。
