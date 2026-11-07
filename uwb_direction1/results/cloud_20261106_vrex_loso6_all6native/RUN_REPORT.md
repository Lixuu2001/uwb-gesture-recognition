# Full LOSO6 fixed DomainBed VREx: all6native

## 2026-11-07：VREx完整六折64生产＋各自fresh64/原生审计已完成；训练结束，全六折私有档案传输中，仍未评分

先核GitHub最新36b35fc95a41e25f3805132a416a8f71fd343655四份权威记录及LOSO协议/报告，大型memory空正文按同commit Gitblob远端SHA精确恢复读取。原64529/start21331595/boot ad409b48-89ce-4704-beee-3658103d5856现为已结束Z；STATE实际completed_folds=[1,2,3,4,5,6]、state=all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once。没有重启/新增训练、重复controller或自动任务。最终主阶段wall31157.163367/CPU31189.589143秒，约8.65/8.66小时，低于各43200秒上限。RESOURCE_LEDGER是最后定期观测（held6/output、completed1..5），不能用它覆盖最终STATE或误称仍有训练；本次明确以最终STATE为完成成本依据。

全部六折各完整64生产和自己fresh64重放、65条epoch记录与全部step日志精确一致，原生first/selected-output/model/BN/optimizer/RNG审计均完成且train/replay/output exit0/无资源stop。更新数每轨迹依次4736/4736/4544/4544/4608/4608；仅各自验证next用户选epoch41/55/19/31/5/11。复用已发布角色/选模通用audit完成held6，与前1..5记录合成覆盖全部6：源6900ID唯一/仅users1..6、受试者及ID严格隔离、三个私有角色副本与源顺序完全一致，64条验证日志规则独立重算一致，不读取held真值/预测来选模型、不冒称独立重算原验证指标。

新增只读all6freeze audit独立核37科学文件/所有frozen输出SHA、六套COMPLETE科学字段、65epoch/全部日志/counter/恰update501前全optimizerreset一次/前500次weight1其后10。首次辅助断言错误地连不同语义state标签也要求相等，实际六折均仅state字段分别training_complete/replay_exact不同；已留错误原因和修复记录，改为各自期望标签及所有其余字段精确比较，一次有界只读重跑通过，没有改科学源或重训。确定性重放不能单独排除共同实现错误。实际训练wall合计15621.136837/CPU15613.786259秒，独立重放wall15441.824002/CPU15433.095525，输出审计wall71.151209/CPU66.822374；主STATE已包含这些，不再相加重复收费。此前MMDwall30865.434969/CPU30891.040083及所有历史检查/归档成本继续另计不抹去。

已按冻结operator/full_archive.py构建全部六折完整模型/独立重放/原始日志/代码/实际样本ID的全档：550659116字节、76块、1980载荷，archive bea0d657323adf3cd984551303bbb08a796be76978eaab03fc225fb5be48d25c，MEMBERS9cf80c64841b2d5ec96cf458d4ee29dacbc793a18e5ecf5ff38d63bd151718ae。构建wall30.366796秒；wrapper只测parentCPU.000803，未单独测child总CPU，不能当完整构建CPU报告。原Drivefolder metadata确认owner-only，当前唯一最多3条完整upload-ID持久-metadata-rawfetch(true,false)-立即物化-SHA流水线执行中；本提交快照仅15/76块通过原始回读，部分索引scope明确不完整，未作整档/allmember验证，不冒称全私有恢复完成。原已保护held1..5完整单折和held6 producer248保持原范围，不重传；原始大数组/runtimebinary不入档案，所有模型/实际ID/原step日志只私有、不入Git。

必须继续现有76块传输，只对已知ID恢复raw，不盲目重复上传；全部块+整档/全部1980载荷SHA通过后，精确发布FULL_PRIVATE_RELEASE并核ref/全部文件，然后先已预登记operator/inference_cost.py固定无标签配对推断成本，再唯一operator/finalize_cohort_once.py全6一次评分。当前SUMMARY/FINALIZE_ONCE及旧held1once均未创建/调用；没有准确率、没有新模型拟合，不按任何单人测试分数决定余折/配置。固定native617612/全3雷达/RMS/B64/64/seed20261029/LOSO6/验证选模及VREx默认10/500/等域BCE/完整原生reset500不改。

完整6花费门槛保持mean+0.5pp/至少4严格改善/最差配对不退超2pp；若有效，先按MMD评分前冻结计划完成等域BCE+同reset/no-variance匹配六折联合对照后分析归因，不直接称原创或跳8确认。原MMD40.6258% vsGAP93.6672%固定失败/非因果诊断、历史混合94.6290%不同预算旧失败均不追改；用户7–8不用于选择，历史全8反复开发探索性、非独立新用户验证。原789/私有Drive/仓库可见性/现有自动频率启用和云环境均不变。目标未达，继续完整档案核验与预登记评分流程。

