# Matched LOSO6 fixed DomainBed MMD: held5producer

## 2026-11-06：held5 完成64轮生产；原独立重放继续，生产产物已私有保护

核对最新4548997ee8fb5bd8f25faf4a65959a73c043746a四份权威记录及冻结LOSO协议/报告；超大memory正文以同提交Git字节匹配远端blob SHA核验。原云环境boot ad409b48-89ce-4704-beee-3658103d5856、phase PID50654/start18218982持续运行，无数据恢复、新训练/重放启动或自动任务变更。

held5生产完成64轮、4608次更新，epoch2仅由验证用户6选择（训练1/2/3/4，测试5）。原生产进程已退出，train阶段exit0、首更新独立原生审计exit0；核验完整64轮journal链/全部更新前缀、最终模型/优化器/全部RNG状态、best_model与选中模型及预测文件SHA。原fresh replay PID57152/start20009626同boot实际存活，观测8/64轮、576更新。生产训练完成不等于独立重放/最终输出审计通过；本次不评分、不改变配置/epoch/seed或六折继续规则。

新增只读producer_archive.py仅保护已退出的完整64轮生产，必须核对原phase正在同折replay、原生产进程身份已退出、原30科学文件及49依赖SHA、生产终止/完整模型/选中模型/journal/前缀；不执行前向、梯度或新重放，不写原模型/日志。实际保护held5生产模型、私有ID/划分、所有轮次/日志/检查点、未评分验证/测试预测、原生源码与初始化索引：8块、51511489字节、241载荷，原私有Drive owner-only/raw fetch/materialize/SHA/整档及每成员校验通过。archive SHA7de6fec7866a11227653c6da8024798048f10f13b867c7cfc960d4389bf737b8，MEMBERS1a5ac1487af64836484d3ca7600986549f055127deec03113d25e33317684150。complete_training_producer=true；complete_modelset/independent_replay/fold_audits/sixfold_cohort均false，无独立自动恢复授权。不得用此生产备份提前调用cohort scorer。无原始大数组进入Git，无已核验分块重传。

held5生产阶段wall2587.385322/CPU2586.213378秒（含首审计）；观测新增阶段累计18245.391336wall/18264.515607CPU秒，原held1成本5022.569200wall/5019.332419CPU另计且保留。新增阶段上限43200wall/CPU秒和RSS12GiB不变，备份完整性检查/构建4.641338wall/4.633900CPU秒及全成员回读成本另记。held1原21文件/未评分状态、通用30科学文件不变；held1完整14块415载荷和held2/3/4各完整13块400载荷保留。旧单折once、VREx、cohort scorer及辅助推断计时均未执行。辅助推断注册已实际绑定4548997，FROZEN SHA5f05da91f9b930226788f5fb06b217a1f9d856c4d2c858b2c282a9385080b4b2，实际注册JSON此次留存，未来full_archive会包含operator目录。

接续先检查真实PID/boot/start与STATE/RESOURCE_LEDGER，继续原held5fresh64/输出审计，再由同控制器执行held6；禁止重复启动或依据单测试人停止其他折。生产备份已存在时不要等价重复创建/上传；没有实质新完成、故障、风险或必须决策时安静。全部六折冻结且原phase退出后，按原full_archive.py完整保护剩余五折与held1组合，原始每成员回读，精确发布FULL_PRIVATE_RELEASE；随后运行已登记的无标签推断计时，再仅cohort finalize_once统一评分。任一传输/执行故障先保留证据、已知ID只回读、未知ID先两轮精确查询后最多一次有记录重试；禁止盲重传/重训。最终分别报告accuracy/present-class macro recall/support/缺类null/用户等权mean/worst/paired差值/严格胜出人数、另列pooled accuracy及全部训练/推断/复现成本。>=0.5pp/>=4胜/最差配对>=-2pp花费门槛不变，正负证据均保留，不追改旧失败。用户7–8不用于选择，八用户历史开发仍属探索性，未声称原创性、独立验证或准确率提升。频率/启用/环境/私有Drive/仓库可见性/原789源码快照均不变，目标未达，接续继续。

