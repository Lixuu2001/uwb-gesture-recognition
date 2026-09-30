# 实施期间补充检索

2026-09-28。主调研见 proposal_v2/31_第一部分新路线_用户校准与多雷达预算识别_调研方案.md。本页只补充实施时核查的原始来源，不将文献的其他数据集结果当成本数据集证据。

- [Cross-Domain Few-Shot Learning With Task-Specific Adapters, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.html)：论文任务含未见类别和未见域。本轮是已知12类上的新用户适配，只提供其中部分动作的支持集；不能混称为全新类别识别。
- [Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition, ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html)：讨论顺序信息获取及联合有用的特征。启示是额外两路可能共同有效，不能只测每次加一路的贪心收益。本轮只有三个固定真实视角，适合枚举明确动作，不需要先上大型强化学习。
- [Active feature acquisition via explainability-driven ranking, ICML 2025](https://proceedings.mlr.press/v267/guney25a.html)：个例特征排序和策略学习已经存在。动态选视角本身不能主张首创；必须赢过固定子集、随机混合和置信度级联，并证明实际处理成本优势。
- [Evaluation of Active Feature Acquisition Methods for Time-varying Feature Settings, JMLR 2025](https://www.jmlr.org/papers/v26/23-1635.html)：离线获取策略评价有明确统计假设。本轮使用完整配对观测离线重放，能够计算固定模型各子集的结果，但不能外推出传感器改变采集方式后的真实世界效果。

检索还出现跨学科少样本类增量原型校准工作（HyCal, CVPR2026）；它属于相邻任务，不据标题将其作为本轮直接方法或声称已完整复现。当前先导刻意从可解释的简单统计模型出发，结果通过停止门槛之后再增加复杂模型对照。
