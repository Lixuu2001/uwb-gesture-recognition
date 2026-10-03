# 频率成分与标签信息：证据边界

2026-10-03。实际读取作者仓库[HaohanWang/HFC README](https://github.com/HaohanWang/HFC/blob/master/README.md)（blob7989543e5139e5924e0686c1a0ab4332fd0f4c50）和[utility/frequencyHelper.py](https://github.com/HaohanWang/HFC/blob/master/utility/frequencyHelper.py)（blob49cb27cc5b8c168473bdc98451bfa998e286795b）。README指向Wang、Wu、Huang、Xing，CVPR2020，High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks，arXiv1905.13545；不是此前推测的ICML。作者展示自然图像高频与语义相关，CNN可能利用它；源码使用二维FFT径向掩码及互补掩码分别重建低/高频图像。

只读作者说明与相关代码，未取得论文全文，不声称全文核查完成。当前可用网络策略未列arxiv/CVF，未绕过网络限制。作者仓库树没有PDF。自然图像CIFAR实验不证明雷达跨用户结果；本研究采用一维距离DCT而非作者二维径向FFT，也不是原论文复现。未复制作者实现，采用SciPy已有DCT与独立余弦公式核验。

上轮DCT21源内诊断说明Left幅度时间差分能量有16.67%被去除，但无法区分动作细节和噪声。本轮只问这部分能否支持跨源用户类别读出及对低频的增量。固定截止21，不改变上轮统一压缩门槛、不扩大截止网格。单独三个视角的幅度，各用低频、高频、混合256维固定随机投影，保留全部90时间帧，匹配线性分类器参数数目。训练标准化与用户类别权重只用五个源训练用户，留一源用户只用于最终诊断；用户1/2无模型处理。

这是既有频带分解和随机投影线性读出的信息诊断，不是原创算法。高频分数高不保证在CNN或多视角系统中互补，也不意味着物理速度信息；高低频在这里沿距离轴。投影/线性表达局限意味着阴性结果不能证明高频只有噪声。若固定门槛通过，仍需另冻结端到端匹配消融、强基线及原八折验证，不能直接宣布研究贡献成立。

## 本轮补充：作者学位论文第2章正文

从作者个人主页仓库直接取得[博士论文PDF](https://github.com/HaohanWang/HaohanWang.github.io/blob/master/file/haohanw_phd_lti_2021.pdf)：Toward Robust Machine Learning by Countering Superficial Features，Haohan Wang，CMU-LTI-21-018，2021年9月。11,464,208字节，Git blob f472e9236aa30806de15b9e42cb55a101ae1311e，SHA256 5c3e1fc67a2a0bbf4e2f962e4c7d41ad8a52268332a039f88e2837514d3129e0。通过允许的codeload GitHub档案下载并核blob；没有绕过被限制目的域。已读取第2章全部正文、表格和图注（印刷页5–16），未逐图视觉审查，也未读全本论文或另取得CVPR正式全文；本段更新前文“只读README/代码”的本轮阅读范围。

与当前诊断直接相关：§2.1采用低/高频互补掩码，明确高频利用不等于样本记忆，高频可能跨训练/验证/测试集预测。§2.2.1表2.1的模型用低或高频训练，却在原图测试，存在输入频带分布差异；不能拿其高频较低分数直接证明我们同频带训练/测试设置的结果。§2.2.2讨论归一化可能放大幅度较小的高频线索，是作者在图像上的假设与证据，不是本雷达机制结论。§2.2.3平滑卷积核的对抗鲁棒性提升伴随clean accuracy下降；因此不能把抗攻击收益等同跨用户收益。§2.2.5用截断SVD作补充检查，不能简单把高频称作随机噪声。

本次54个固定读出显示Right高频可读出31.7323%，但混合未超过低频；Left高频21.5707%也高于12类均匀输出的8.33%参考值，但未到预设25%信息门槛。这里只是描述性比较，未执行标签置换显著性检验。门槛失败不是“信息为零”的检验。没有依据这些结果调参或删改既定门槛。
