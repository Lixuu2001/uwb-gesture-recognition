# Full LOSO6 fixed DomainBed VREx: preparation

## 2026-11-06：关闭完整六折MMD后落实新VREx完整LOSO实现，先预登记，尚未训练

最新bbb6ac659e829edc8ce13a7d67f9a57aafbacd39已冻结MMD完整六人一次统一结果：40.6258% vs原GAP93.6672%，平均-53.0414pp/0胜，确切gamma1七核配置关闭，无补救调参。全部MMD6模型/独立64重放/原生审计及私有恢复链保留，原held1once锁仍未消费、原科学文件未改。MMD总wall30865.434969/CPU30891.040083秒另计，后续不能清零。

依据MMD评分前6975049695db4f306446d58bf85c1b8f48231e24已冻结的NEXT_VREX_LOSO_DESIGN_BEFORE_SCORE，实际建立新/workspace/uwb-vrex-loso6-20261106：通用loader、完整nativeengine、逐折train/first/freshreplay/output worker、六折串行累计资源监督/ownedPID停止、全部checkpoint/RNG/journal、完整私有归档、无标签配对推断和唯一cohort scorer，共37科学文件。不是旧held1singlefold conditional package，旧VREx22源和run未激活状态不修改/调用。静态AST/六折角色及更新数、12个原GAP模型/输出hash核验通过；当前仅实现，未运行新UWB训练或声称后台已启动。

文献为Krueger等ICML2021/PMLR139:5815–5826原文及MIT DomainBed b93c22a1：论文Eq8风险sum与声明实现risk mean有区别，BCE替换CE/原生AdamWLookahead完整reset/pooledforward-BN也明确为迁移，不称完整作者图像benchmark复现或原创。实际VREx engine与既有评分前固定字节17be97571844dd032907d9f9760a621c098793c8994c3aa4167fc60c881b3d28相同。lambda10、anneal500、weight1直到已完成499次、从500起weight10，update501前完整原生优化器reset一次；全计数/resetEvents/fastmoment/slowbuffers/RNG恢复检查。完整617612/三雷达90x567/RMS/B64/64/seed20261029/1thread不变，测试h1..6、验证h%6+1、其余四人训练，仅对应训练/验证选checkpoint。每折生产64+独立fresh64+全部步/轮次/原生输出审计，原GAP六折强基线只复用不重训、不降低。

主VREx较GAP同时改变等域损失均值与reset，不能直接归因variance。若完整主六折达到固定花费门槛，必须先完成评分前固定等域BCE+同reset500、varianceweight0的联合六折对照（同init/roles/budget，独立10h上限、全6producer64和原生审计），VREx均值还须严格超过联合对照才谈variance贡献/另规划8外层确认。不拆分归因mean/reset，不凭模块组合称创新。原历史混合94.6290%保留旧完整门槛失败/64+100预算不同。

本次先逐文件精确发布/readback并force=false冻结源manifest。之后仅跑全6角色固定native合成检查（实际pinnedclass、独立解析梯度/gradcheck、真正小optimizer501次、native617612全部梯度/BN、slow6/disk7/全部RNG、tail28），玩具ownedPID停止测试和固定生成B64九步/每臂含resetfixture计时。唯一资源门槛1.25*已测MMD6 CPU30891.040083423*max(1,VREx/native ratio)+600<=43200，ratio取两臂九步总wall/CPU比的较大者；不得以真实数据pilot或反复计时/缩骨干救预算。新主六折阶段wall/CPU各43200秒、RSS12GiB，一模型活跃；生产无自动重试。预登记全部通过、未训练初始state+全部6私有ID/role完整Drive raw/allmember回读、实际精确release后，才能启动唯一六折过程；此提交尚未启动。全部6模型/预测/审计及私有完整回读先冻结，才一次统一评分，不能任何单人分数决定继续他折。

固定指标accuracy与present-class macro/support/缺类null分开，用户等权mean/worst、配对差值/严格胜出，pooledaccuracy与训练/推断/复现/归档成本另列；>=+0.5pp/>=4胜/最差配对>=-2pp/tol1e-12花费门槛不代表任何效应证据有无，正负均留痕。用户7–8不用于选择，全部八人历史开发探索性，非独立新用户/已验证原创贡献。现有自动任务频率/启用/环境、Drive私有、仓库可见性与原789快照不变。目标未达，继续上述具体原生核验与可运行实验。

