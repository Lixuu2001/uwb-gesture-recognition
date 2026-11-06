# Matched LOSO6 fixed DomainBed MMD: fullprivate

## 2026-11-06：完整MMD六折训练/重放/原生审计完成；全模型私有组合冻结，尚未评分

先核对最新9fa07a5e1828b6679bed8a860e51fed7cced174e的四份权威记录、冻结LOSO协议/报告及真实进程。原phase50654/start18218982、boot ad409b48-89ce-4704-beee-3658103d5856已正常退出（Z），最终STATE明确新增held2..6完成，ALL_SIX_PREDICTIONS_FROZEN已写出。复核五折各64轮生产/独立fresh64、全部65轮记录和全部更新日志逐项精确一致；各train/replay/output exit0、无资源停止。held1原64+fresh64/415载荷完整档案复用，原21科学文件和旧单折once均未更改/调用。通用30科学文件和原生49依赖SHA不变。每折更新数为h1/2=4736、h3/4=4544、h5/6=4608，选中epoch分别1/1/2/2/2/1，仅用对应验证人。全部六人预测ID顺序、12类float32输出、角色/样本ID隔离及原GAP固定模型/预测SHA已盲核验，未打开真值。

原full_archive.py构建剩余五折完整恢复包，含所有模型/优化器/全部RNG、私有ID/角色、每步/轮次/journal、选中/验证/测试未评分预测、原生源码、原初始化私有索引及辅助推断协议/实际注册：62块、454825959字节、1672载荷；archive SHA1da2ef9a8ee7e7222223c5d16cddf8b44a5fb5d6c9d49286ebfd571cc7cdc884，MEMBERSb045a3bd9d605392e0ecf9b9d4e8ad6190b5c8d095fa8b1468a0fa6e4ccd347a。原私有Drive owner-only/rawfetch/materialize/SHA/整档及每成员通过。结合原held1完整14块101419672字节415载荷索引并重新核验整档/全部成员，形成完整六折恢复链；剩余五折包不是单独包含held1模型的档案，恢复必须按组合索引。所有已存在各折/生产/部分备份保持原范围，不重传或改结论。原始数组/实际样本ID/模型不进入Git。

上传最多3条完整分块流水线，持久UP_ID/metadata/rawdownload/SHA/回读记录后才成功；已知ID失败只再次原始回读，未知ID必须两轮父目录+全局精确名称完整空查询、传播间隔及一次落盘限额标记才可追加一次上传，原错误全部保留，无盲重传/重训。传输恢复汇总及实际代码留痕。此处完全是归档传输，后台模型训练已结束，不能声称又启动训练。

新增五折实际wall25842.865769/CPU25871.707664秒（约7.18小时），小于43200秒上限；原held1另计5022.569200/5019.332419秒，总模型训练+独立复现wall30865.434969/CPU30891.040083秒（约8.58小时），未清零。归档/全成员回读和随后推断计时另列，不与模型拟合或纯推断混淆。

本次全包/原held1/盲schema及成本信息先精确发布并逐文件回读，force=false；之后才把真实commit、剩余五折archive SHA及原cohort scorer SHA绑定P/FULL_PRIVATE_RELEASE.json。此提交时尚未打开六折真值、未运行辅助推断、未创建cohort once锁；也没有调用旧held1 once或激活旧VREx单折脚本。接着仅运行已注册operator/inference_cost.py --run固定一遍无标签配对推断计时（B64/1线程/固定warmup/交替臂顺序，逐位匹配冻结预测、模型缓冲区不变，300秒上限），然后原operator/finalize_cohort_once.py统一计算全部六人。不得因单个人的分数决定是否执行其他折或改参数。

固定指标分别为accuracy、present-class macro recall/support/缺类null、用户等权mean/worst、paired差值/严格胜出人数，另命名pooled accuracy及完整训练/推断/复现成本。冻结>=0.5pp/>=4胜/最差配对>=-2pp花费门槛不变，不等同任何正负证据有无；旧失败不追改。用户7–8不用于选择，全部八人历史开发仍探索性，不声称独立新用户验证/原创贡献。原GAP93.6672%保持强对照，历史混合94.6290%仅不同预算强参考。

在MMD六折评分前还核查Krueger等ICML2021 REx主论文的方法/理论限制/DomainBed实验节与已固定MIT实现；仅记为后续候选，旧VREx22源文件及未激活状态保留。预先固定完整六人LOSO的lambda10/anneal500/全617612方案、等域均值+reset联合对照计划及有界预算，禁止依据MMD逐人分数调整这些值；若MMD通过则不激活替代路线。若MMD完整六折关闭，推进独立新六折实现/预登记，绝非旧held1 once条件脚本。论文及真实准确率证据与原创性分开，不以已发表为由放弃实际实验。现有自动任务频率/启用/云环境、私有Drive、仓库可见性与原789源码快照均不变。目标未达；先完成当前唯一cohort评分，再按真实结果推进。

