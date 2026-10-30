# 2026-10-30 — TS‑TCC完整HAR默认多通道适配已实现并预登记，真实训练未启动

本轮按94364d1远端四文件和实际云端环境核查，无活动旧训练/审计，不重复下载；原数据/cache/source6900 IDs与maps存在。新增web工具可读出版方HTML https://ijcai.org/proceedings/2021/324，已读Methods3.1–3.3、Implementation4.2与evaluation5.1–5.3/Table1–3显示正文；PDF重定向失败、公式未解码、图未看，不称全文复现。它支持原标准线性评估与跨域微调两个预登记读出，不因先例而停止实际准确率检验。

/workspace/uwb-tstcc-pilot-20261030 已实现完整HAR默认internal32/64/128、TC100/depth4/heads4/6steps、batch128、每阶段40轮：SSL→linear→finetune→scratch共160epoch、5760updates、737280训练出现。encoder247500+TC313096参数，567channels/90frames/12classes/latent13为形状适配。透明修复NumPy2变长分段shuffle和原pat[0,warp]首通道广播为pat[:,warp]全通道同一时间置乱，显式轴向，不用原min-dim heuristic。保留原codeGaussianmean2/sigma1.1弱增益、strongjitter.8/maxseg8exclusive/空分段，与论文prose/MHAR10差异明示。固定训练增强一次、流式float32与完整修正参考257样本/跨chunk以及最终NumPy RNG逐位一致。四阶段原作者一步更新对照、checkpoint恢复后两步双模型/双优化器/scheduler/全部RNG逐位通过；原linear保留train-mode dropout/BN统计更新仅冻结参数，FT与scratch完整训练。额外FT读出依据论文5.3在真实数据前固定，非看分后补救；初始三阶段合成记录保留。

四项守卫测试通过；完整默认32次合成update/val测时成本估计2785.263668秒，低于43200CPU/wall，maxRSS686012KiB。cost/test独立一线程并行执行，可能放大计时，未用真实UWB值/标签/拟合。固定seed20261029、4700源3–6训练(每epoch4608/丢92)、1000user2验证、1200held1诊断；无7–8标签。完整40末轮选模、原ReduceLROnPlateau用val2 CE，仅修正val drop_last使1000全部保留。三个分类器全部验证/诊断logits/ID/预测先冻结再读held1标签。任一预定臂至少超过最强同held1参考98.666667% .5pp才允许另行source确认，否则关闭确切配置，不扫seed/增强/epoch/宽度。准确率与原创证据分开，八用户历史开发下探索性。

协议 research/tstcc_pilot_20261030/PROTOCOL.json SHA024d044ce1a619526b8072672e1b76bf9623748ddc57dbcab0d937fe2be43d3a，71项冻结科学/源码/依赖/前任务证据。源码/许可未知的作者10blob不复制入Git，已有哈希/获取helper可恢复。训练/完成审计/首步验证/守卫已实现，必须远端逐字读回后才一次启动。真实训练尚未启动，不能称后台运行。独立训练43100wall/43000–43100CPU；独立完整重放审计7200预算，核全部5760步/737280训练ID出现、6900RMS、4700增强、120验证epoch和6600final logits；需完成私有rawreadback才接受gate。保留每阶段initial/first/rollinglast及独立10/20/30/40检查点含双模型/双optimizer/scheduler/所有RNG/IDlog，无自动retry/resume。

下一步：确认本预登记远端精确读回，核run01不存在且旧TS2Vec保持complete/auditpass/privateverified，然后 python launch.py --run 一次。核真实首步模型/优化器/128ID后记录真实PID。仅观察已有训练；完成后 python launch.py --audit，保留完整检查点/日志/样本ID并私有备份原始取回SHA验证。目标仍未达，继续任务启用，原789源码快照未动，实际数组/模型/样本ID值不入Git。

