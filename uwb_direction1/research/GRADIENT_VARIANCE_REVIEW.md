# 源用户梯度方差：文献与计算核准备

2026-10-04。仅准备，不是新UWB训练/分类结果，不是研究贡献已验证。

## 阅读依据与范围

Ramé, Dancette, Cord, **Fishr: Invariant Gradient Variances for Out-of-distribution Generalization**, ICML 2022，作者仓库 https://github.com/alexrame/fishr ，论文入口 https://arxiv.org/abs/2109.02934 。已读作者README全文（含摘要、实验表与模型选择说明）和实现指定区间，固定提交7b8fdf1e0b15226ded9b58efd37698e74e616ab7及blob/SHA见GRADIENT_VARIANCE_REVIEW_RECEIPT.json。主论文全文未读；当前云端网络受限，未声称已完成全文核查。仓库LICENSE头为GPL-3.0；未把作者代码复制进项目。

Fishr匹配各训练域的逐样本损失梯度方差，与特征均值对齐、用户平均风险DRO、SAM邻域扰动不同。作者DomainBed代码只对classifier取逐样本梯度，求每域中心化的总体方差（分母N），再与等域平均方差的均方距离比较；它不是直接精确Hessian。README中的视觉数据集结论不能作为UWB有效证据。README同时报告Oracle选模（涉及测试域），本项目禁止采用这种选模方式。

实现细节：DomainBed对逐样本CE求和后取BatchGrad，因此不得把梯度又除以batch size；将weight和bias全部坐标拼接后取均方，不能按层先分别等权平均。EMA输出除以固定1−ema，这不是按时间1−ema^t的通常偏差校正；先前EMA状态detach，但当前批方差需保留对特征与分类器的梯度。默认lambda1000、anneal1500步、ema.95是该实现默认值，不应照搬为本数据最优值。其anneal转换时重置Adam，未来比较必须明确控制额外重置的影响并保存EMA/计数器/优化器完整恢复状态。

发现README对两个ColoredMNIST变体的文字解释互换；代码fishr_offdiagonal计算完整协方差，fishr_notcentered跳过中心化。ColoredMNIST还有与DomainBed不同的loss缩放、参数范数和sum/mean归一化，不能混用默认权重或声称两条实现相同。

## 独立数学核与实际验证

本项目新文件tools/linear_gradient_variance.py从softmax CE解析式实现末层线性分类器逐样本梯度：r_i=softmax(Wz_i+b)−onehot(y_i)，gW_i=r_i z_i^T，gb_i=r_i。对每域分别中心化求总体方差，再平均域间/坐标平方差。这是已知数学计算，不是原创方法；仅覆盖线性末层，不等价于任意非线性classifier的全部Fishr参数梯度。没有EMA、退火、训练器或自动选择超参数。

check_linear_gradient_variance.py使用逐样本autograd CE作为独立oracle，实际通过float64梯度行、惩罚标量、对特征/权重/偏置的高阶导数一致性，以及有限差分gradcheck。另验证样本乱序与整批重复不变、单域为零、单样本域拒绝。见GRADIENT_VARIANCE_CORE_TESTS.json。只用合成样本，未读取UWB诊断/测试标签，没有新的泛化成绩。

## 继续前须解决的可否决问题

旧源端训练后期误差为零，梯度方差可能整体接近零；SAM固定实验的轻微增益未过门槛，并不能证明存在可利用的跨用户梯度方差失配。先在源训练角色上预注册一个有类内置换/有限样本噪声对照的诊断，检验梯度方差差异是否超出采样噪声，并记录绝对量级与估计稳定性。不要读取held用户来选择梯度坐标、时间点或阈值，也不要以事后大方差挑用户。

原始源码中精确Fishr/Fish/gradient variance/梯度方差/IRM/VREx标记搜索未命中仅是去重线索，不是新颖性证明；旧DANN、跨用户均值约束、DRO等不得换名包装。当前尚未冻结新UWB诊断协议或启动新训练。若机制诊断不支持，停止扩展；即使支持，仍须来源清晰、同源角色/计算量/容量对照及后续同协议八折，普通Fishr本身不能作为原创贡献。
