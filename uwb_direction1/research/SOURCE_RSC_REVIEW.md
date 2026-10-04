# RSC 源端编码器训练：已知方法的固定对照筛查

Huang, Wang, Xing, Huang, *Self-Challenging Improves Cross-Domain Generalization*, ECCV2020，arXiv2007.02454。作者仓库 https://github.com/DeLightCMU/RSC/tree/bf6d280c5d74910f009ea8963c59167252659666 的 README 和 Domain_Generalization/models/resnet.py 第65–158行已读。作者当前代码有后续课程式调整，随机空间/通道挑战，不是本轮固定向量规则。没有取得或通读论文全文，不把 README 结果或“停止调参”建议作为本实验实证。

另核查 DomainBed https://github.com/facebookresearch/DomainBed/tree/b93c22a1cfc3b2428398272c1a116c8de1f4139e 的 algorithms.py 第1248–1310行及 hparams_registry.py 默认配置：特征和批次丢弃比例均1/3，正确类别logit的有符号梯度、行分位数、预测概率下降的批分位数、逻辑或掩码。本实现独立按数学规则编写；没有复制作者实现。作者BSD2-Clause、DomainBed MIT许可已读。固定全向量规则只作为已知CV方法，不能称原创。

既有附加描述没有通过强控制，下一假设是训练期挑战容易依赖的表示是否促使编码器使用更广泛线索。当前1728维是共享CNN三雷达特征连接，非物理轨迹。辅助选掩码禁用分类头dropout，最终反传保持原dropout；这与论文/DomainBed分类头配置不完全相同，不能称精确复现。批选依据来自各臂各自模型，随机控制只在该状态保持逐样本移除数量及选择规则；两条训练轨迹不会保持相同逐样本选择。它控制特征定向性，不完全控制整个训练动力学。严格分位数相等项不保留，零梯度可使被选样本全屏蔽；不能看到结果后修规则。

合成数据测试独立NumPy分位数、同数量随机屏蔽、辅助RNG不推进、普通更新与完整断点逐位一致。真实数据只在远端预注册回读后计时和训练；训练/验证/诊断角色与旧源基线一致，只训练标签选掩码，主固定tail13–24轮，验证CE最优仅次指标。普通基线首epoch必须复现逐位相同。RSC对普通tail至少+0.5pp，对random+0.25pp，且不低于已有SAMtail和CNN岭读出；每项≥4/6胜且每用户不低于−1pp。失败关闭本组掩码/比例/seed/window网格。即使通过也不是原创或独立八折确认。

只做有界GitHub仓库检索及当前项目关键词排重，未发现直接本轮RSC项；检索缺失不证明新颖性。完整文献范围和文件哈希见 SOURCE_RSC_REVIEW_RECEIPT.json。
