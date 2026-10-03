# 二阶统计候选的依据与排重（2026-10-03）

当前仅完成阅读，尚未实现/冻结/运行二阶分类实验，没有新识别结果或创新主张。

## 已否决的重新提议

本次先查了“三雷达训练、单雷达部署”是否遗漏。原不可变c735e94的OOF_PRIVILEGED_KD_PLAN、oof_privileged_results/RESULTS、TEACHER_FREE_CONTROL_PLAN和teacher_necessity_results/MECHANISM_NOTE表明已做过：三视角OOF71.90%、单视角OOF72.14%、无教师全局平滑74.77%；三视角OOF相对两控制−0.23/−2.87pp。teacher_uniform74.61%、label_uniform74.65%；原教师训练样本T=2真类概率0.9842，目标较接近标签。这些数值是原报告记载，本次没有重训/重算，不与当前源六人macro指标排序。普通KD和这组OOF策略保持不扩展；不能重新命名为创新。

本次查询原789文件树，没有covar/riemann/second/spd/bilinear/scatter/rocket命名命中；这只是文件名筛查，**不是整个原源码不存在协方差计算的证明**。接续须查相关模型内容，尤其已完成的静态矩/时序矩实现，再冻结方法。

## 可核查的另一类CV表示

[Li等，CVPR2018作者实现fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV)对应 *Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization*，其前作是ICCV2017 *Is Second-order Information Helpful for Large-scale Visual Recognition?*。

本次完整读取README和src/representation/MPNCOV.py，精确blob见SECOND_ORDER_REVIEW_RECEIPT.json；未读主文全文，未复现作者ImageNet结果。实现：可选1×1通道降维，空间位置中心化协方差，迹归一化后Newton–Schulz矩阵平方根迭代，乘回sqrt(trace)，上三角展开；包含手写backward。不是逐元素开方，也不是普通均值或单通道方差。原源码标注all rights reserved，本项目未复制该文件；若实现应独立表达数学定义、注明来源并处理适用许可。

[pyRiemann](https://github.com/pyRiemann/pyRiemann)的README和完整pyriemann/tangentspace.py也已读。README援引Barachant等2012/2013的SPD协方差分类及切空间方法；TangentSpace.fit从训练矩阵估计参考均值，transform默认使用已拟合参考，tsupdate=True会另用预测批矩阵估计参考。后者是额外测试时适配假设，不能默认带入本实验。此库以EEG/多变量信号及遥感为主要例子，不是UWB-Gestures的有效性证据；论文全文未读。

这两条路线不可混称：矩阵平方根归一化协方差池化不等同仿射不变Riemannian切空间；后者也不等同直接对特征逐元素log。

## 下一步应回答的具体问题

现有CNN以共享48通道卷积和3×4自适应平均网格表示各雷达。旧静态矩主要保留逐位置/通道的均值、方差或时序矩；它不自动保留卷积通道间联合协方差。可检验池化时丢失的跨通道二阶相关是否提供跨用户分类补充，但不能依据这一直觉宣称已有瓶颈。

优先只在旧源3–8四训练/一验证/一诊断协议内，使用固定源训练检查点的末层特征，分别比较均值/对角方差、未归一化完整协方差、矩阵平方根协方差；保持单雷达内部统计，不假设雷达间逐帧同步。若先用固定低维投影减成本，应在标签评分前固定投影和维度，报告信息压缩与参数量差异；强均值/方差控制和相同正则选择不可省略。

先做数值PSD/零输入/独立协方差及平方根核查和真实CPU成本探针，再冻结源内判定；所有标准化/参考矩阵/读出仅训练用户拟合，验证用户选预定超参数，诊断用户仅最终报告。不用全数据参考、不默认tsupdate，不以旧八折测试值调矩阵维度/收缩/窗口。即使源先导通过，也只支持进一步对照，最终仍须完整同协议强基线、八折消融及原创贡献核查。现阶段无新模型训练运行。
