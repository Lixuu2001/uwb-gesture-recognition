# Full LOSO6 fixed DomainBed VREx: all6private

## 2026-11-07：VREx完整六折全部模型/重放/原生审计的76块1980载荷私有原始回读与整档SHA通过，评分前完整冻结

以最新b9b8c251e25dc22df7b402d0f1071f7931bfc089为父精确发布。本次完整恢复包550659116字节、76块、1980载荷，archive bea0d657323adf3cd984551303bbb08a796be76978eaab03fc225fb5be48d25c、MEMBERS9cf80c64841b2d5ec96cf458d4ee29dacbc793a18e5ecf5ff38d63bd151718ae。76个唯一Drive ID/上传响应/原始回读收据一一对应，无错误/额外上传/已知ID重传；最多3条完整流水线。原folder最终metadata再次确认owner-only，逐块metadata/rawfetch(download_raw_file=true,include_base64=false)/立即物化/SHA后，整档及1981项含MEMBERS/1980载荷hash全部通过。complete_modelset_verified=true、complete_all_six=true，范围确含全部6折模型及独立重放/原始日志/代码/实际ID；原始大数组、source_maps和runtimebinaries仍为外部恢复链，不冒称本档案单独包含整个运行环境。无新训练/推断/测试真值配对。

完整持久方法索引full_backup/CLOUD_DOWNLOAD_INDEX.json已齐；新增COMPOSITE_RECOVERY_INDEX明确方法全6、独立旧GAP baseline链与原始86文件/163块/2141981100字节迁移索引的组合关系及SHA。baseline held2..6先前15块405载荷恢复/原生输出审计保留；held1继续从held6旧档案引用的upstream author-core私有链恢复，不能称当前新方法档案包括所有baseline模型。原始cache/maps重建必须按冻结protocol代码及原789源码验证目标SHA；本次没有重做原始数据迁移/缓存重建/基线训练。所有样本ID和模型原值仅在私有档案，Git只含代码、metadata、实际索引/哈希与结果。

整档解包/全部payload回读校验wall4.343533/CPU4.341403秒；76块首上传响应持久到最后rawSHA收据跨度1630.766131秒，排除初上传延迟、包含与Git发布/只读审计重叠的等待，不把它当纯网络性能。此前构建wall30.366796秒，child总CPU未单独测量，parentCPU.000803不可当总构建CPU。完整模型主阶段最终STATE wall31157.163367/CPU31189.589143秒（约8.65/8.66小时，均<12小时），包含6生产64/各自fresh64/原生first/output审计；训练wall合计15621.136837、重放15441.824002、输出审计71.151209秒，不能与主阶段再重复相加。原MMD全部wall30865.434969/CPU30891.040083秒及历史检查/备份/故障成本完整保留。

六折训练/验证/测试subject与样本ID严格隔离，独立初始native617612/全3radars/RMS/B64/64/seed20261029；只按各自验证next用户选择epoch41/55/19/31/5/11，所有64轮轨迹保留。完整native64+自己的fresh64/65epoch/全部step/optimizer/reset500/RNG及原生输出审计已通过；独立角色/验证选模规则与全部counter/weight/reset意图审计保留，确定性重放不能单独排除共同实现错误。辅助state-label断言修复只在外部只读audit，科学37文件与模型不改，原失误/修复记录已保留于上次提交。

本次SCORING_RELEASE_INTENT在任何真值配对之前锁定完整档案/37科学文件/once scorer/固定推断protocol哈希及原门槛。精确发布并逐文件回读/ref force=false核验后，本地FULL_PRIVATE_RELEASE绑定该实际提交；按原operator/inference_cost.py做一次无标签12个配对推断（B64/1线程/固定warmup/300秒wallCPU、冻结logits和BN不变），随后才operator/finalize_cohort_once.py全6统一一次评分。当前SUMMARY、INFERENCE_ONCE、FINALIZE_ONCE均未创建，旧held1once未调用；不先看任何一人分数、调配置/seed/门槛或恢复任何已完成训练。

结果须逐人accuracy与present-class macro recall、12类support/缺类null、用户等权mean/worst/paired delta/严格胜人数、样本加权pooled accuracy分开报告。固定花费门槛仍mean提升>=0.5pp、至少4/6严格改善、最差配对退化不超2pp；门槛不是正负证据存在的定义。若有效，先按MMD评分前冻结计划执行等域BCE+同reset/no-variance匹配六折联合对照再归因，不直接称原创或跳8确认；若未过，关闭确切配置，不据测试救参。原MMD40.6258% vsGAP93.6672%固定失败/非因果诊断、旧混合94.6290%不同预算/旧失败保留；用户7–8不参与新方法选择，历史全8反复开发探索性，不能称独立新用户验证。原789/仓库可见性/Drive私有/自动任务频率启用/云环境不变，研究目标未达，本提交仍无新测试准确率。

