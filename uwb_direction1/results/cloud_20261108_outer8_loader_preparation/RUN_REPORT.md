# Real label-free outer8 input preparation: loader_preparation

## 2026-11-08：八人外层LOSO无标签真实加载器完成，未启动新训练

最新状态：已完成真实9300样本的八人subject/ID/RMS按需加载准备，h1..8测试、验证h%8+1、其他6人训练，ID和人严格隔离。既有六人6900输入与冻结原生RMS缓存逐元素完全一致；新增用户7/8各1200输入仅作无标签有限性/索引审计。未打开手势truth数组、未访问旧source_ids.npz的labels键、未新拟合/模型推断/评分。原六人IDs保留原矛盾排除，7/8无新增过滤，不从未知标签推断类别support。

全量9300缓存需1,898,316,128字节，开始时空闲1,583,353,856字节；采用只读原数组+最多64样本按需signed RMS，未新建大缓存、未删除科学文件。全输入审计outer墙钟12.206249s/CPU12.605846s、峰值3,586,956KiB；这不是训练或模型推断耗时估计。SWA原论文方法/理论限制/实验/附录及作者固定411b2fc版本已核查；作者SGD/epoch末采样与既有AdamWLookahead/密集33–64控制不同，保留后者的正向描述性证据，不倒改为当前主方法成功。作者组件生成数据检查通过；物化多一末尾换行已记录、恢复精确Git blob并重新检查，旧检查/成本保留。

所有旧MMD/VREx/Fishr/IRM/GroupDRO/SWAD失败结论、SWAD六折冻结54源码、SUMMARY及消耗once锁不变；旧MMD试验once仍未调用。八人方案仍是探索性，历史反复暴露全部8人，不能称独立新用户验证。未改变自动任务ID6abc995e1620819195fcff77dbd81118的启用/每小时频率/继承同云环境。

下一步必须推进独立预登记的前瞻八人worker、角色标签防火墙、训练内有界成本测量和显式资源/存储准入。方法须在读取7/8手势标签前固定，不能按新标签选窗/seed/LR。完整八折需匹配的6源人基线与方法，不能用旧4源人的六折权重/分数替代；新的阶段预算不得抹去旧MMD12小时阶段及第一折成本。现有数据加载器准备不是新方法准入、不是后台训练已启动；不重复六折训练/控制/once评分，不以非原创为由无限停在抽象否定。

详情：results/cloud_20261108_outer8_loader_preparation/RUN_REPORT.md；代码和冻结协议：research/cloud_20261108_outer8_label_free_preparation/。当前无新训练进程，目标仍未达成。


Sources: [SWA paper](https://arxiv.org/abs/1803.05407v3), [pinned BSD author code](https://github.com/timgaripov/swa/tree/411b2fcad59bec60c6c9eb1eb19ab906540e5ea2). Detailed INPUT_CHECK/COST and source hashes accompany this report. This is preparation evidence, with no new accuracy or originality claim.
