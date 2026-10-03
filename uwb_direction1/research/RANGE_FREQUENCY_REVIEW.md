# 频率成分与标签信息：证据边界

2026-10-03。实际读取作者仓库[HaohanWang/HFC README](https://github.com/HaohanWang/HFC/blob/master/README.md)（blob7989543e5139e5924e0686c1a0ab4332fd0f4c50）和[utility/frequencyHelper.py](https://github.com/HaohanWang/HFC/blob/master/utility/frequencyHelper.py)（blob49cb27cc5b8c168473bdc98451bfa998e286795b）。README指向Wang、Wu、Huang、Xing，CVPR2020，High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks，arXiv1905.13545；不是此前推测的ICML。作者展示自然图像高频与语义相关，CNN可能利用它；源码使用二维FFT径向掩码及互补掩码分别重建低/高频图像。

只读作者说明与相关代码，未取得论文全文，不声称全文核查完成。当前可用网络策略未列arxiv/CVF，未绕过网络限制。作者仓库树没有PDF。自然图像CIFAR实验不证明雷达跨用户结果；本研究采用一维距离DCT而非作者二维径向FFT，也不是原论文复现。未复制作者实现，采用SciPy已有DCT与独立余弦公式核验。

上轮DCT21源内诊断说明Left幅度时间差分能量有16.67%被去除，但无法区分动作细节和噪声。本轮只问这部分能否支持跨源用户类别读出及对低频的增量。固定截止21，不改变上轮统一压缩门槛、不扩大截止网格。单独三个视角的幅度，各用低频、高频、混合256维固定随机投影，保留全部90时间帧，匹配线性分类器参数数目。训练标准化与用户类别权重只用五个源训练用户，留一源用户只用于最终诊断；用户1/2无模型处理。

这是既有频带分解和随机投影线性读出的信息诊断，不是原创算法。高频分数高不保证在CNN或多视角系统中互补，也不意味着物理速度信息；高低频在这里沿距离轴。投影/线性表达局限意味着阴性结果不能证明高频只有噪声。若固定门槛通过，仍需另冻结端到端匹配消融、强基线及原八折验证，不能直接宣布研究贡献成立。
