# Full LOSO6 fixed DomainBed VREx: all6result

## 2026-11-07：VREx完整六人LOSO已一次统一评分：92.8444% vs GAP93.6672%，-0.8227pp，3/6改善；固定配置未过门槛并关闭

评分严格在全部六折64生产+各自fresh64/原生审计、角色/ID与验证选模审计、76块550659116字节1980载荷私有原始回读/整档全成员SHA及精确发布037d3c0877e75af8211b4ba5b269c4db8127a5e9之后。FULL_PRIVATE_RELEASE绑定该实际提交，已登记无标签12次配对推断成功（wall40.110690/CPU40.094307秒含startup，所有冻结logits逐位一致，模型/BN不变），随后原operator/finalize_cohort_once.py唯一一次完整真值配对成功（wall6.976469/CPU7.382373秒含preflight）。FINALIZE_ONCE.lock已消费、SUMMARY和FINAL_RESULT_STATE已完成，finalizer_invocations=1，旧held1once调用0；不能再启动/评分/重新计时。原P/STATE保留评分前冻结快照，完成判读以新FINAL_RESULT_STATE和SUMMARY为准，不覆盖科学快照。

逐测试用户GAP→VREx accuracy/present-class macro recall及差：h1 98.4167→96.5000%（-1.9167pp）；h2 98.7000→98.9000%（+0.2000pp）；h3 90.0000→90.5000%（+0.5000pp）；h4 94.2500→93.9167%（-0.3333pp）；h5 96.0000→96.2500%（+0.2500pp）；h6 84.6364→81.0000%（-3.6364pp）。用户等权mean93.6671717→92.8444444%，平均配对-0.8227273pp、3/6严格胜；最差用户84.6364→81.0000%，最差配对h6退3.6363636pp。样本加权pooled accuracy93.6521739→92.8405797%，与用户等权均值分开。三个正向用户与全部负向结果同时保留，不把未过门槛说成完全无正向证据。

accuracy是所有样本正确率；present-class macro recall仅对真support>0类别等权。用户2缺0-based10/11、用户6缺3，其余全12类，每个现有类各100条，因此本数据逐人两值数值相同但定义不等同；缺失类recall=null、预测到缺失类仍算错误。额外精确分数审计仅从已评分聚合support/recall重建整数正确数，不重新读取测试labels/predictions，核对两指标定义、所有null、均值/严格胜数/原门槛与冻结scorer完全一致。固定均值>=+0.5pp/至少4严格胜/最差配对不退超2pp三个条件均未满足，关闭确切nativeGAP64/3radars/RMS/seed20261029/VREx10-500配置，不能根据测试救lambda/anneal/seed、改门槛或重开旧单折。只在有效时计划的联合匹配对照本轮未激活，没有新训练。

完整逐人accuracy与present-recall、每类support/缺类及recall、成本和图见本报告同目录report_artifacts/FINAL_REPORT.md、PER_USER.csv、PER_CLASS_SUPPORT_AND_RECALL.csv、COST_REPORT.json、LOSO_COMPARISON.svg。图表只派生已冻结SUMMARY，不读取原测试标签或拟合；公开render_cohort_report源码与summary/产物哈希回执。模型/实际样本ID/未评分预测和原始日志全部在私有完整档案，索引及组合恢复图已公开保留；不把外部raw/cache/runtime说成已含于单档。

实际主阶段训练+重放/原生审计最终wall31157.163367/CPU31189.589143秒，约8.65/8.66小时；训练wall15621.136837、独立重放15441.824002、输出审计71.151209秒已包含于主总值，不能重复相加。76块首响应至最后rawSHA收据跨度1630.766131秒含与其他只读/Git工作重叠；整档/1980载荷hash校验wall4.343533/CPU4.341403秒，构建child总CPU未单独测量。此前MMD全部wall30865.434969/CPU30891.040083秒与所有已有成本完整保留。旧GAP只恢复核SHA与输出，没有重训/降基线；旧混合94.6290%不同预算及原失败结论仍原样，不能追认成功。

准确率提升证据：该固定方法完整六人开发LOSO平均下降，只有3个用户描述性改善，未达到本轮主目标。原创贡献证据：未验证；VREx为ICML2021已发表方法的DomainBed默认组件迁移，同native骨干/数据/64轮预算，但等域BCE和完整optimizerreset500相对global-BCE GAP有声明的差异。本负结果不归因于单一variance机制，也不声称精确重放足以排除所有共同实现错误。用户7–8未用于新方法选择，全部历史八人反复开发仍探索性，不称独立新用户验证或八人确认。

研究目标未达，维持现有自动任务频率/启用与原云环境，继续有文献依据的不同确切配置/方法的实现和真实验证。下一项先对照已发表作者默认实现与历史缩小/改写版本，冻结全六人协议/seed/配置和有界CPU成本后再启动，不能仅凭不能即时声称创新停止；不重复本VREx/MMD/旧单折脚本，不扫测试集。原789源码/旧科学文件和旧失败结论/仓库可见性/Drive私有均保持。

