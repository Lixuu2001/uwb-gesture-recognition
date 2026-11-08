# Published SWAD native full implementation: core_preparation

## 2026-11-08 GroupDRO全6固定负结果已关闭；下一SWAD完整native核心/恢复生成检查通过，尚无新UWB训练

父fcdb8298bfce62e224aafe53865e1f92e0db65be已精确21文件回读：GroupDROeta.01/full617612六折93.5654040404% vs GAP93.6671717172%，-0.1017676768pp、3胜3负、最差paired-1.8pp；77私有块/2010载荷、12组盲推断和唯一联合评分完成，原配置关闭/once消耗，不重训/重评/换eta或seed，不启动仅通过主门槛才规定的eta0对照。最差用户+0.2727pp及其他局部正效果保留，但整体无准确率提升。完整逐人accuracy/presentmacro/support/null缺类/pooled/成本与错误修复见results/cloud_20261108_groupdro_loso6_all6_result/FINAL_REPORT.md。

按评分前c8cebdc8记录的失败分支，推进已发表SWAD LossValley默认3/6/.3完整native组件。通过GitHub解析作者khanrc/swad不可变commit252190eb881169e736beccf51af5caaa5c5cf666，7文件blob/SHA验证。读取完整LossValley/AveragedModel/update_bn、trainer接线和registry/config/README引用许可；未读完论文全文，不称完整作者benchmark或原创。官方默认pretrained/freeze_bn=True，valley死后提前停训；本项目将保留native617612/全三路RMS/B64/64/seed20261029/nativeBCE+AdamWLookahead/原生BN，以独立验证用户损失选平均区间并完成64，固定同轨迹tail33..64，未来明确源训练ID-only BN校准和selected-GAP BN-only匹配输出。BN/完整6fold runner仍未实现，不能用旧best/final伪造密集平均，也不为恢复而重训基线。

实际实现CPU适配（只重定向作者最终cuda）和可恢复纯Tensor/基本类型的完整segment/tail/queues/threshold/count/event状态。三组手工推导标量角落、实际64epoch标量tail33..64/拒绝二次finalize和NaN验证通过；完整617612生成模型逐参数独立float64密集平均最大差1.6391277313232422e-7，与不收集平均的原生8次更新/model/optimizer/RNG逐位一致；原生slow6/disk7/exact8含全部队列/平均/RNG/mode及finalize恢复通过。不是新UWB准确率结果或64+fresh64真实六折结果。生成open firewall阻止实际workspace .npy/.npz和raw restored目录读取，无真实数组/手势标签/样本ID访问。

元数据inline曾缺布尔赋值导致SyntaxError，修为持久显式dict，未启动测试/计时。首次生成native8步发现官方np.mean阈值为NumPy标量，weights_only=True存盘加载失败；保留失败源码/stdout/stderr，用同IEEE64数值的Pythonfloat规范化状态边界，未改作者算术/默认值，也未禁用weights_only或新增unsafe globals。失败尝试尚未做性能profile、无UWB数据访问，其8次B8生成native更新成本未单测，不报0。修复后完整生成suite wall11.507317244017031/CPU11.503240485000001秒通过。

只做一次事前固定8步/臂（丢3warmup、交替顺序）B64生成core profile；固定1.25*31189.589142517*max(1,SWAD/native_wall,SWAD/nativeCPU)+1200给42087.66081777077秒(11.6910h)<12h。1200秒为前瞻BN/journal余量，此处不是完整BN/执行器成本或真实fit放行；不重测同代码寻找过关值、不缩模型/数据/B64/64轮/seed。下一步具体补source-only BN校准/全部6fold loader-worker/native reference/averager原子journal/真实64+fresh64生成生命周期/ownedPID12h监督与角色审计、精确全源码及6初始状态私有raw release后才可一个新phase。目前不存在/workspace/uwb-swad-loso6-20261108，也没有SWAD训练/initial保险/完整runner，不宣称后台已启动。

历史SWAD24/seed2026/users3..8/legacyGN/cosine/decay.001失败结论完整保留；新fullnative64/BN/seed20261029/users1..6/原生constantAdamWLookahead不同，仍是已发表组件迁移而非新算法。已发表方法可作有价值强基线，准确率证据与原创证据分开；不据旧缩小实现抽象否定完整设置，也不包装已知组合为创新。用户7-8不用于新选择，历史8人研究探索性；原789源码、数据和自动任务频率/启用状态/云环境均不改，目标未达继续。以下为历史。

