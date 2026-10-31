# 2026-10-31 — 固定滤波分支同参数消融原型通过合成检查；真实source6训练持续，不新增诊断选择

先通过GitHub读取最新ce6b442699c52336a33685056bc532440d1d034f四主记录，确认同一cloud运行环境/数据/实际进程，不重复下载或启动。只读观测2026-10-31T18:19:08.544985+00:00：controller59236、job2/3 59237/59238、train59239/59240正常；用户2已72epoch/5328updates、用户3已75epoch/5325updates，4–6排队，failed0、完整新fold0，1465冻结SHA再核通过。本折诊断标签均未用于评分，无source6最终准确率。原100epoch/B64/seed20261029/Adam/选模/源划分/门槛全部不变，继续已登记训练→审计队列，不提前切方法或根据中途指标调参。

本轮具体推进未来的可归因性消融实现：/workspace/uwb-hinception-fixed-branch-control-20261031，Git research/hinception_fixed_branch_control_20261031/。重新核对H-Inception作者README/既有完整源码及InceptionTime作者README/完整inception.py；后者固定commit952d115e83fb3a66d75c5858c702f8dd5eeb18b7、3原文件blob/SHA与GPL保留。2022 DOI10.1109/BIGDATA55660.2022.10020496/2020 DOI10.1007/s10618-020-00710-y支持固定分支与普通Inception架构区别，但不是新发现论文、全PDF/图表复核或创新证据；历史已有InceptionTime调研，不能换名算新方法。

**zero_fixed_responses**仅在原17固定Conv+ReLU后加一个注册、无权重zero层；固定卷积仍计算，保留第一concat145维、所有851434total/601006trainable/50tensors/102Adam与初始化。full模式直接原nativeauthor，无额外改写。合成seed7/B2证明：full对原作者初始+3updates所有native model/Adam/BN/metrics/globalRNG逐位一致；full/zero初始所有状态和变量shape完全一致；第一模块128learned特征完全一致、只有17固定response为0；17固定kernels在3updates后不变；zero状态恢复next2所有native状态+模块modes精确，注册层.keras safe reload的weights/optimizer/preds精确，非法mode拒绝。真实UWB值/标签/fit为0，没有正式CPU测时/实际ablation配置或accuracy结果。

这个控制问“固定分支的输入相关信息是否有用”，**不是删除结构、原版InceptionTime或推理加速**；17个下游BN affine维仍可学习常量，保留的固定计算+额外zero不等于严格墙时一致。它也不单独隔离RMS/znorm/预算/优化器/多雷达贡献。现有LocalCNN/GAP为固定pipeline参考而非此因果控制。未来只有source6完整audit/私有恢复/原gate完成后，才允许另行固定真实消融cost/配置/seed/协议/runner/auditor，并远端逐字回读后fit；当前原型不授权提前训练或7–8标签选择。全部source探索性，不能称新贡献/八折改进。

本次只提交代码、源码引用、合成实际检查及真实运行观测/记忆，不含raw/maps/真实ID值/模型/二进制，不改789快照和冻结训练文件。下一步仍等现有5fold双预测全部冻结及完整审计→aggregate，再使用已备妥complete_preservation_tools/完整模型集私有回读后发真实source6比较。目标未达，保持既有自动接续，无重复任务。

