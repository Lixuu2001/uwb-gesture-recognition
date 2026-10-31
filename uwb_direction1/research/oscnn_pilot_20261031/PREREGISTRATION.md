# 作者多变量OS-CNN完整网络：固定CPU100准确率实验

作者Wensi-Tang/OS-CNN commit f70626d3c4aad70a76ba7b4745f556df01044279（MIT），普通multivariate notebook的OS_CNN，不混合residual/ensemble/per-var版本。原模型/structure builder不修改；完整第一层8*128*C预算、第二层5*128*256+2*256*128，C567/L90/12类，1870464可训练参数/3736380含固定mask/14活跃Adamstates。few_shot=False原分类head保留。wrapper min(L//4,89)=22实际核1,2,3,5,7,11,13,17,19，卷积局部RF1–38，GAP仍汇总90帧，不声称局部全尺度覆盖90或输入截断。已发表强基线迁移，准确率与原创贡献证据分开，不称创新。

ResNet CPU100已经完整7400步审计pass并按注册门槛close，8私有chunks/81payload全部读回重组验证通过后才进行本候选成本筛查。原1500轮ResNet未做，不用CPU100负结果否定它。OSCNN成本policy先于计时，固定唯一完整网络/B16/seed20261029/monitor50，按2000(constructor)→500(author demo)→100(explicitCPU适配)顺序选首个<=43200。一次8train/4grad-enabled eval合成probe，估2000=288270.600015、500=72170.999791均超，100=14544.439732秒通过。独立600wall/590–600CPU进程exit0/no timeout4.377930wall/4.375241CPU。此估计不是时长保证；100是明确预算适配，不是2000/500或Torch1.1原论文复现，也不把2000/500判为科学负结果。

固定seed20261029/B16/100完整轮/1CPU线程/deterministic，每轮4700全记录/294批末12、29400更新/470000出现，无earlystop。Adam.001/betas.9,.999/eps1e-8/decay0/no clip。原ReduceLROnPlateau(min,.5,patience50,relthreshold1e-4,min_lr.0001)喂每轮最后一批loss。每50轮先eval shuffled trainloader再eval sequential user2，再train并覆写原periodic model，最终last；不是valbest或额外选epoch。sourcefit的requiresgrad scalar warning只用last.detach()兼容，其值和状态经完整合成fit对照已证明等价。epochmean仅日志，不拿来调度或选模。

固定全90×567有符号real record/view RMS，连续转置为567×90，无resize/augmentation。source6900IDs，train3–6=4700/val2=1000/held1诊断=1200，ID/maps/rawSHA绑定。标签仅角色内转换int64，12类用既有taxonomy，不合并train+test拟合类数/onehot。原TEST监视改为仅user2，held1两个最终ID/logits/pred均落盘哈希及finalpredictionstate后才索引标签；7–8标签不用。

原生global Torch DataLoader train shuffle=True/numworkers0/drop_lastFalse（periodic train-eval也shuffle），val/diag顺序B16，保留global RNG消耗。第三TensorDataset字段是实际IDs，只观察不改变训练。每步保存实际ID/loss/顺序，periodic所有714个evalbatch实际ID也记录并重放。源weight.data=weight*mask每个forward清零、Adam后hole regrow、eval也改权重均保留；不静默替换gradientmask/稀疏网络。保存initial/first/rollinglast/finalprediction全model/Adam/BN/scheduler/mode/PythonNumPyTorchRNG/identity/sampler和原periodic/final rawstate；不自动中途resume/retry。first证明只核immutablefirst/global sampler16IDs/状态，不新增raw或label/realforwardfit。

真实runner、独立全29400逐步审计、firstproof、备份和controller均具体实现；五项具体入口/分批/原生global loader及evalmask/checkpoint/finalprediction重放检查、三项controller守卫已过，仅合成。旧source core3updates/restore2与NumPyconv、engine完整作者fit2epoch/22updates/periodiceval/checkpoint/allRNG、NumPyCE/restoreepoch/IDobserver/identity守卫证据绑定。真实43100wall/43000–43100CPU，完整审计14400秒。auditor还核6900RMS、全部训练IDs/470000出现、100epoch源scheduler、714evalbatches及evalweightmutation、initial/first/last/finalprediction全状态、raw源权重和最终2200logits/pred逐位。macro归约仅预有1e-12，状态/输出不放宽。

本条写入时新run不存在，未读OSCNN真实UWB值/标签/fit；须本次代码/协议/哈希/准备资料远端预登记逐字读回后才创建marker并仅一次launch。缺marker/hash错误/前任务未完整close/已有run均拒绝。实际首步通过即私有保险，controller只等已有训练成功→完整重放一次→本地完整包，无Drive/Git或训练重启能力。agent最终owner-only metadata/rawfetch(download_raw_file=true,include_base64=false)/即物化SHA/max3pipeline/重组全成员后发布。异常留trace/TERM先修，不当科学失败。

双预测freeze后单次held1macro相同reference .9866666666666667，提升>=.005（1e-12容差）才另注册source确认，否则关闭确切CPU100并转下一方法。不得据诊断/7–8label改seed/epoch/batch/kernel/width/normalization救援。源因八用户历史开发探索性，非独立test/旧8fold/原创贡献。paper仅相关primary方法/设置读，非全17页/图。目标未达，既有小时接续保持；Git仅本次代码、配置、真实结果、记忆，原789/原数组/maps/真实模型/实际IDs不入Git，不依赖Windows。
