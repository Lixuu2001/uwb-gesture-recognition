# Actual IRM native LOSO6: running

## 2026-11-07：IRM六折真实阶段已启动；原进程92152第一折训练中，独立真实首步审计及3块私有中途检查点回读通过，尚无新准确率

以最新记录为准：用户模板中的MMD“尚未评分/未补五折”是历史状态。原MMD六折40.6258%、VREx92.8444%、Fishr93.5737%均已完成审计及一次统一评分并按各自冻结条件关闭，GAP六折93.6672%与旧混合模型94.6290%的不同预算/原门槛失败结论不变，不重训/重评分/修改原科学文件。当前推进已文献核查和完整预登记的IRM迁移配置，准确率证据与原创贡献分列，研究目标仍未达到。

源提交9c149bc773ebffd9abb01cd07638eaadea43eedf精确70文件、初态保险提交4a9ca2a37f1680b9f6ac445a840318ca1b79eaf5精确19文件核验后才写REAL_RELEASE并启动唯一phase。实际主PID92152、bootad409b48-89ce-4704-beee-3658103d5856、startticks29520187，启动Unix1794064627.8829243；本次观察held1真实生产已提交epoch26/64、1924/4736更新，worker PID92154仍活跃，全部45项冻结科学源码字节不变。STATE根文件在首折完成前缺失属控制器生命周期预期，不能据此判断没有运行或重复启动。LAUNCH_ACTUAL的“updates尚未观察”是启动瞬间快照，当前ACTIVE_OBSERVATION才是运行观察；不覆盖原快照。

实际真实首步模型/原生优化器/slow/RNG状态精确独立重放已通过；另一路独立dummy-scale autograd核查全部617612参数clipped梯度，最大绝对差7.450580596923828e-9，训练标签only，不是只用同实现精确重放排除共同错误。只核验首步不等于完整64+fresh64已通过。私有中途快照已核验实际epoch15、1110提交更新的全native/journal/prefix和5条完整未提交suffix，已发生reset_events=[500]，原生新optimizer count610及全部fast step610；捕获1115条记录均满足warmup1→weight100、globalBCE/penalty/loss/counter和reset501规则。此证据确认真正跨过退火边界，不是仅500轮生成fixture，也不基于测试分数选择。

只读一致性快照没有暂停/改变原进程或读取UWB/gesture truth：3块20942074字节、93payload/tar94成员，archiveSHAed4aae79ef6d78894ba9c6eed2c88824820bc23b4e794682103f3e902e1c295d，私有原目录metadata唯一owner/shared=false，raw fetch download_raw_file=true/include_base64=false、立即物化size/SHA、最多3整条pipeline、所有块/整档/成员回读通过。权重/实际ID/数组保持私有不入Git，Git仅来源/配置/审计/索引。该保险是单一中途native checkpoint及现存journal，不是完整折、轨迹或六折；恢复索引同时链接原始86文件/163块、既有GAP和all6初态保险。先核验当前真实PID+boot+startticks，健康原任务继续；失活时保留最新checkpoint/optimizer/slow/RNG/journal/suffix和全部已用/未知成本后审查恢复，禁止自动从初态重训或重置预算。当前只有第一折快照获私有保险，后续更新尚未在此快照中。

本次观察主训练累计wall1036.145秒、CPU1037.808秒、RSS1701616KiB，新增全六折阶段实际wall/CPU各43200秒和12GiB硬上限不变；预估11.6378h仅容量估计，不是已用或承诺完成。原absolute及script容量估计失败、显式calibrationV2缘由和未改变硬上限均保留。快照捕获wall0.095633/CPU0.095602秒，allmember核验见READBACK_COST；其与运行墙钟重叠，不加两次，网络/完整运维CPU未测不填0。旧MMD wall30865.434969089023/CPU30891.040083422995（其中h1另显5022.569200421014/5019.332418999999，不再相加），VREx31157.163366941997/31189.589142517，Fishr29559.94331930898/29599.386025808秒成本全部另保留。

下一次自动接续仅先检查此原phase身份/资源/日志/native产物，不新建自动任务或重复worker。原phase顺序h1..6各64生产→fresh64逐步native重放→无测试truth输出；完成全部六折后才完整全模型/轨迹/预测/角色/选模/原生审计/私有raw-allmember及exact release冻结，再一次统一cohort评分。禁止调用旧MMD held1 once或提前单人评分，不能用任何单折分数决定剩余折。主划分仍testh1..6、valh%6+1、其他四训，完整GAP617612/all3 signedRMS/B64/64/seed20261029/globalBCE；IRM100/anneal500/warm1、native全optimizer reset500为方法偏差已登记。正结果先同globalBCE/reset500/no-IRM匹配六折对照归因，负则关闭确切配置不救参。报告accuracy与present-class macro/support缺类null、equal-user mean/worst/paired/wins及成本；固定+.5pp/4胜/最差>=-2pp不变；用户7–8不用选择，历史八人探索性非独立验证，不声称原创或提升。789原源码、仓库可见性、现有自动任务频率/启用和云环境保持不变。

