# Matched LOSO6 fixed DomainBed MMD: held5complete

## 2026-11-06：held5 完整64+fresh64及原生审计通过；原控制器继续最后held6

核对最新31e516f35cb3d15897e7ef54d060b7b19cc2d6e1的四份权威记录和冻结六人LOSO协议/报告；超大memory空正文用相同提交Git字节匹配远端blob SHA核验。原云环境boot ad409b48-89ce-4704-beee-3658103d5856和原phase PID50654/start18218982持续运行，无数据恢复、基线重训、新实验/重放/自动任务启动。

held5已完成64轮生产与独立新鲜64轮重放，各4608更新，全部65份纯轮次记录及全部更新日志精确一致；最终模型/优化器/全部RNG状态一致，选中模型及验证/测试输入输出原生审计通过。三个阶段均exit0且无资源停止。epoch2只由验证用户6选择，训练1/2/3/4、测试5；不读held真值、不评分。原held1科学21文件/未评分状态、通用科学30文件及原生49依赖均核验不变；旧单折once、VREx、六折评分和辅助推断计时未调用。

复用已发布completed_fold_archive.py，完整held5模型/初始与first/双槽/terminal/selected、优化器/全部RNG、私有样本ID/划分、所有更新/轮次/journal、原生源码及未评分预测已完成私有原始回读：13块、93193258字节、400载荷；archive SHAcfe39ef23c4f97a3d58fad27406511603c8cf7f04f66cc1b8e93249a87cb591d，MEMBERSf81ffa1d241318fcde58cb671ffd939fa2d69448a47801c97f4a8687f315066b。complete_single_fold=true但complete_sixfold_cohort=false。先前held5生产专用8块241载荷索引仍按原范围保留，不改成完整重放证据。held1完整14块415载荷及held2/3/4各13块400载荷仍有效。私有owner-only/rawfetch/materialize/SHA/整档及全部成员通过，最多3条完整分块流水线；原始数组和实际ID/模型未进入Git。

原phase完成新增held2..5，当前最后held6实际model PID58597/start20270838同boot存活；首更新独立原生审计exit0，观测训练51/64、3672更新。尚需完成held6生产64/独立fresh64/输出审计；没有依据任何测试人分数改变后续执行。held5训练wall2587.385322/CPU2586.213378秒，独立重放增量2597.193458/2596.298929秒，输出原生审计增量12.036498/11.436739秒，折总5196.617769/5193.949046秒。至held5新增阶段累计20516.146496wall/20538.127797CPU秒；先前held1另计5022.569200wall/5019.332419CPU秒，未清零或遗漏。新增阶段wall/CPU43200秒、RSS12GiB上限不变。备份完整性检查/构建5.251463wall/5.243356CPU秒及全成员回读成本另记，不当成模型推断时间。

接续先核对原PID/boot/start及STATE/RESOURCE_LEDGER；继续原held6，不新开重复模型，不重复创建/上传已有完整备份，无实质新变化时安静。全部六折模型/预测冻结且原phase退出后，按原operator/full_archive.py构建剩余五折完整模型集并与原held1私有索引组成完整六折恢复链，原始整档/每成员回读后精确发布FULL_PRIVATE_RELEASE。之后运行已注册无标签inference_cost.py --run记录固定一遍、B64/1线程、单批warmup另记、奇偶交替臂顺序的推断成本；模型/缓冲区须不变、logits须与冻结预测逐位一致，不读取held真值。注册实际commit4548997，INFERENCE_FROZEN SHA5f05da91f9b930226788f5fb06b217a1f9d856c4d2c858b2c282a9385080b4b2，运行中原phase已验证会在打开模型/数组前阻止计时。不得把仅12秒输出审计称作纯推断成本。

最后只调用已登记cohort finalize_once统一评分；禁止原held1单折once或据此自动启动VREx。报告每人accuracy与present-class macro recall、support/缺类null、用户等权mean/worst/paired差值/严格胜出人数，另列pooled accuracy和完整训练/推断/复现成本。冻结均值>=0.5pp、>=4胜、最差配对>=-2pp花费门槛不变；正负结果均报告，门槛不等同任何证据有无，旧失败不追改。用户7–8不用于方法选择，八人历史反复开发明确探索性；方法选择后另规划同条件完整八人外层LOSO。未声称准确率提升、原创贡献或独立验证成功。现有频率/启用/云环境/私有Drive/仓库可见性/原789源码快照保持，目标未达，接续继续。

