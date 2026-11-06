# Matched LOSO6 fixed DomainBed MMD: held6producer

## 2026-11-06：最后held6生产64轮完成并私有保护；原独立重放继续，六折仍未评分

本轮先完成held5完整64+fresh64/全部原生审计及13块400载荷私有回读，已精确发布57a92a3d812ab34bf979ea3620bbb3a1855afed6。原phase PID50654/start18218982、boot ad409b48-89ce-4704-beee-3658103d5856持续运行，随后观察到held6生产完成64轮、4608更新，原控制器自行进入独立重放；没有新建/重启训练或自动任务。

held6生产stage exit0、首更新独立原生审计exit0，epoch1只由验证用户1选择（训练2/3/4/5、测试6）。复用已发布producer_archive.py，只读核验已退出生产的最终模型/优化器/全部RNG、选中模型与best_model、64轮journal链/全部4608更新前缀及未评分预测文件SHA。独立重放实际PID60014/start20530226同boot存活，观测4/64轮、288更新。生产完成不等于完整折审计通过；完整六折冻结、私有组合归档和评分仍待完成。

held6生产模型/优化器/全部RNG、私有ID/划分、全部日志/轮次/检查点、验证/测试未评分预测、源码及初始化索引私有owner-only/rawfetch/materialize/SHA/整档及每成员通过：8块、51808067字节、241载荷，archive SHAaa629905e22018649b9098fda72ddf6e5de82323b814594a1d0b96e743307453，MEMBERS0c4573250c661177890252d4042f97bef986e7bc12e4dd7289c5fd257ec69a29。范围为生产端，complete_training_producer=true，complete_modelset/independent_replay/fold_audits/sixfold_cohort均false；没有自动独立恢复授权。原held1完整14块415载荷、held2..5各完整13块400载荷及早前部分备份保留各自原范围，不追改旧结论。无原始数组、实际样本ID或模型进入Git，无原始雷达数据重下。

held6生产阶段wall2593.870456/CPU2592.800238秒；观测新增阶段累计23295.780422wall/23321.227263CPU秒，原held1成本5022.569200wall/5019.332419CPU秒另计且保留。新增阶段上限仍43200wall/CPU秒、RSS12GiB；备份构建4.613138wall/4.607869CPU秒及全成员回读成本另记。原held1科学21文件及通用科学30文件不变；旧单折once、VREx、cohort scorer和辅助推断计时均未调用。

接续先核对原phase/PID/boot/start与STATE/RESOURCE_LEDGER，只继续当前held6独立fresh64及最终输出审计；禁止重复新模型、重复生产备份或单折测试反馈。原phase退出并写出ALL_SIX_PREDICTIONS_FROZEN、各stage exit0后，运行原operator/full_archive.py保护剩余五折完整模型集，和原held1私有完整索引组成六折恢复链；严格owner-only/raw/每成员校验，精确发布FULL_PRIVATE_RELEASE。再运行已登记、无真值的inference_cost.py --run记录每人每臂固定一遍B64/1线程推断；注册commit4548997、FROZEN SHA5f05da91f9b930226788f5fb06b217a1f9d856c4d2c858b2c282a9385080b4b2。最后仅原cohort finalize_once统一评分，绝不用旧held1 once或由其激活VREx。

最终分别报告accuracy/present-class macro recall/support/缺类null/用户等权mean/worst/paired差值/严格胜出人数、单列pooled accuracy及完整训练/推断/复现成本。冻结均值>=0.5pp、>=4胜、最差配对>=-2pp花费门槛不变，任何正负证据均报告且不追改旧失败。用户7–8不用于选择，八用户历史开发明确探索性；方法选择后另规划同条件八人外层LOSO。无准确率提升/原创性/独立验证成功声明。现有任务频率/启用/云环境、私有Drive、仓库可见性和原789源码快照不变。目标未达，接续继续；没有实质新完成、失败、阻塞或决策需求时安静，不为发状态而等价重跑或上传。

