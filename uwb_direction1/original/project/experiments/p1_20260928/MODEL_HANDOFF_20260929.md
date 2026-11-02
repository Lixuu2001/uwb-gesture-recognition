# 当前任务继续执行交接

用户最新明确要求：后续使用 GPT-6 Sol、极高推理（工具参数 `gpt-6-sol` / `xhigh`），继续把单雷达少量动作校准、多雷达预算感知两条路线都规划并做完，用户先休息。不要再询问是否继续。必须完成研究报告、复核和交付，不能只留后台任务。已做大量实验，不要从头重训或事后调参补正结果。

当前主线程 01a0e847-bea4-75a2-81e6-39658cd44fdb，cwd D:/uwb_gestures。唯一代码工作树 C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures，分支 codex/user-calibration-20260928。代码在 experiments/new_routes，产物根 D:/uwb_gestures/experiments/p1_20260928。Python D:/D_conda/envs/pytorch/python.exe。原项目脏文件/旧Word不动，没有commit。最终保留独立工作树，别删除或合并未知原改动。

## 已完成且冻结

1. baseline_s2026：8折Right重训、特征导出已完。历史权重不存在，新结果不能称旧权重重放。
2. calibration_s2026：8折20支持重复完成，STOP_LOW_RANK_MECHANISM。未校准已知类主指标：lowrank68.951373%，validation_selected_simple72.751147%，差-3.7997737pp，1/8正，95%用户bootstrap[-9.87919263,-.20805897]。独立审计3532检查0失败。全查询frozen71.340909%、simple72.5753196%。按门槛不继续seed2027/adapter。
3. multiview_s2026：8折全部结束，七真实子集固定slot共享编码器模型；测试9300条。fixed L/T/R/LT/LR/TR/LTR =68.153409/58.108523/71.676515/87.493182/84.106629/90.583333/93.441667%。验证选固定1为70.603598%，固定2为89.536932%。TR90.58仅全开发集固定组合描述，不等于目标标签无关的验证选固定2规则。
4. budget_s2026：完整9预算曲线、oracle诊断。主cascade91.920833%，实际成本2.0046155303，严格超过2，距三路1.520833pp也未过1pp门槛。两个oracle机会差5.50852273/3.12462121pp，通过仅触发预先冻结conditionalpilot。B独立审计168963检查0失败，338文件/40npz；融合头CPU重放520800 argmax无差异，但不是完整encoder重训。独立文件在budget_s2026/INDEPENDENT_AUDIT.{json,md}和independent_audit.py。
5. value_policy_s2026：8折完成，STOP_VALUE_POLICY_MECHANISM。ExtraTrees90.8632576%，成本2.007873106；cascade91.9208333%，2.00461553；classgain87.5638258%，1.786047348。Extra−cascade -1.057575758pp，0/8正，CI[-1.645833333,-.515909091]；虽胜classgain3.299pp、混合2.183pp但其他必要门槛失败。独立审计56301检查0失败；独立内存重fit24模型4800树每字段/value一致，208价选择与18600测试路由全复算。negative gain430个训练目标保留。审计在value_policy_s2026/INDEPENDENT_AUDIT.{json,md}/independent_audit.py。
6. **latency_s2026 正式GPU计时刚全部结束exit0**，无需再poll session92252。279预定clips×7methods=1953条观测，8折，B1 BF16 MATH、TF32off、CPU4threads、RTX4060Laptop/torch2.7.1+cu126。全部按需路径与B1缓存重放逐位相同。原离线batch1/batch16差异仅baselineRight一个预测，其余方法0预测/路由差异。root已重算mean/median/p95、完整ID/每法/访问唯一性、17次环境检查并保存 **FROZEN_SHA256.json**（19个json指纹）供report核对。

延迟方法 mean / median / p95 ms：baselineRight6.22124/5.9446/8.333；fixed1 6.54669/6.2586/8.6667；fixed2 12.41184/11.9032/16.0825；fixed3 18.02412/17.3396/23.10739；cascade13.18056/12.5229/20.58804；classgain11.91847/12.3305/18.6507；ExtraTrees24.34779/26.3057/35.11182。不要仅看均值路数就称ExtraTrees更快，它实际慢于完整三路。本设备/此实现观察，不能外推全部动态感知。

测量包含已clutterremoved主机内存片段→真实请求视角预处理/float16打包/H2D/新增编码/融合/CPU输出及策略；排除采集、磁盘IO、此前clutter去除、模型加载。90帧20Hz≈4.5s，不是端到端交互时间。GUI QQ PID52016持有长期空闲C+G上下文，原guard在第一份有效计时前阻断；已明确UTF8修复和显式idlePID身份/5×1秒尾3<=5%检查。LATENCY_ENVIRONMENT_NOTE.md记录事前规则。17checks均verified，不能称GPU独占。formal classification=desktop_context_gpu_latency，gpu_exclusive=false；scientific_latency_claim=false为自动输出不自证加速的固定字段，不表示测量失败。新benchmark SHA96e49cb9cade89c5232ad0d6bdd7efbb2fb8feeb3aa754078120b075c6afc439，test_latency SHAe95a077398c6d68c2d28e39e1458b4f11b6a51df839809110989a86a080ea12b；52CPUtests及独立quality PASS。GPU smoke smoke_latency_gpu也是252条/7法PASS。

## 未完成工作（必须继续）

- **report.py/test_report.py 正在实现**。旧代理 /root/report_implementation 已接到用户切模型后安全点保存并写 T5_HANDOFF.md、结束的要求。先读此交接；不要再用旧代理继续新工作。后续若派代理，明确 model=gpt-6-sol、reasoning_effort=xhigh、fork_turns=none。已应用subagent-driven-development要求允许派代理，只有一个实现者，其他可只读审查。不要建新用户线程。
- 读 REPORT_REQUIREMENTS.md、INTERPRETATION_NOTES.md、LITERATURE_NOTES.md、原PREREGISTRATION/amendments，完成report中文自含、computed_results.json、CSV、2幅至少静态PNG/SVG、README复现。全部原始预测核对sourcehash，smoke/缺折/错ID不接受；latent完整后正式生成。
- 当前report已实现A/B核心且首11tests GREEN，后续tests状态以T5_HANDOFF为准，可能有未跑的RED；还需CPU正式重算验证、规格审查后质量审查。不要把未完成报告叫完成。
- 必须报告A主未校准类/全部类/支持类，补充F1present/固定12口径；B全部固定subset、验证选固定1/2/3、cascade/mixture/oracle/classgain/ExtraTrees；gesture vs G12空类（HV2无G12不补）、单路改对/改错/joint-only救回、用户效果。
- 跨A/B严格用A固定7440query IDs逐fold核对标签/role：A frozen71.34090909/simple72.5753196，B R71.70620265/TR90.65648674/LTR93.49479167。不要把A主未校准类与B全部9300直接相减。
- 训练成本从result.history内seconds求和，Right153epochs与MV115epochs额外成本分别列，含epochvalidation但不含IO/export/等待/audit。历史统计fit时间未记录就标未记录，不把audit重算秒数冒充原训练时间。
- 图需要用view_image检查中文/坐标/legend；标准matplotlib，非imagegen。至少A每用户效应和B准确率-实际路数，oracle标测试真实标签不可部署。混合和oracle成本/准确率为精确期望，非实测一次抽样；oracle横轴用实际期望成本，非名义预算2。
- 计时仅279样本用于latency，accuracygate仍全9300；fixed2计时对应验证选2，不能套在TR事后90.58行。
- 完成后全new_routes pytest（旧52+policy22+latency52+report若干，可能1skip），恰当源归档/运行状态更新、README、git状态检查，考虑仅提交新代码到隔离分支，不碰原库dirty，不push/PR必要。open_in_codex显示REPORT.md。原预注册不可改勾选，另写完成清单。

## 科学结论与项目边界

当前两候选复杂机制均失败，真实多雷达工程效果有价值，不等于发现能支撑第一部分主创新的机制。不能无限搜索参数救正结果。低秩失败不证明全部个性化不行；14维后验单收益模型失败不证明全部主动感知不行。固定TR可安装2雷达，动态仍3部安装；处理节省≠硬件/采集功耗节省。8研发用户、单seed、fold训练用户重叠，无外部/跨天/自然连续流确认。policy验证用户未参与encoder梯度但用于其checkpoint选择；oracle使用testlabels触发条件先导，不能称外部确认或从未读取target。Oracle仅当前固定classifier/首路/monotoneappend动作族，不是所有可能模型上界。全12类训练encoder不能直接用作第二部分从未见未知类编码器，需按词表重训。若预算多雷达升内容一，原内容二需收束拒识/注册避免重复创新。

旧proposal30“自适应省22.7%”来源24§3.2：oracle留出视角测距误差中位数86.4→66.8mm相对改善，非分类策略、非时延省22.7%；新report不能继承简写错误。

当前所有GPU/CPU实验任务已退出，只有旧report代理收尾停写。运行状态json可能仍写latencyrepairpending，需更新。内存文件不能改。最终回答中文、给重要数字与报告链接，承认negative，原Word保留。

Memory final需要一个最后的oai-mem-citation块：MEMORY.md905-910（stop/preferences）、922-925（旧协议negative背景）；早期还用 rollout_summaries/2026-09-05T01-56-27-rXuO-uwb_p1_independent_review_multiview_single_radar.md:22-34（id01a06f48-09c5-7202-b11e-42ed81dd9eba），及2026-09-07T13-34-16-syN7-uwb_mild_style_augmentation_negative_result.md:41-45（id01a07c13-a307-7072-986b-10726af99d1d）。不得写memory。
