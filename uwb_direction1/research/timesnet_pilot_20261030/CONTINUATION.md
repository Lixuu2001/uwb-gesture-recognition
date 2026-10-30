# 2026-10-30 — TimesNet完整作者手势preset真实源内协议已预登记；远端读回前不启动

先读a34ea717远端四文件、核云端运行/已有进程与路径，无活动旧训练，不重复下载。上一TS‑TCC已完整结束/130哈希全重放/16块完整独立私有恢复verified，并按原门槛关闭；不重开。继续已通过source3步/restore2步/独立FFT/CPU成本的TimesNet当前作者UWaveGestureLibrary完整preset，2433260参数、32/64/e2/top3/6kernels/b16、RAdam.001/decay0/clip4、30epochcap/patience10、seed20261029。不是泛CLI512宽度的手工缩小，不依据真实TimesNet结果选preset。新/workspace/uwb-timesnet-pilot-20261030已实现真实runner、原sourceEarlyStopping和最佳checkpoint选择、完整重放审计、首步检查、私有备份与等待现有训练的后处理控制器。

唯一固定源协议：用户3–6训练4700、2验证1000、1诊断1200，6900source IDs/90×567RMS原哈希不变，无7–8标签。训练每轮294批全部4700/末批12、上限8820步/141000出现，dedicatedGenerator(seed+epoch)作为明示采样适配。原作者TEST用于validation的调用不照搬：仅user2准确率选best，原ties保存later并重置counter/严格下降10次早停，无scheduler。实际epochs可因原patience10早停，非任意缩短预算。保存initial/first/best/last完整model/RAdam/stopper/全部RNG/sampler及原checkpoint.pth和全部实际训练ID/epoch日志。

保留模型固有batch-context Fourier周期排名；预登记validation(seed+101)/held1(seed+102)仅按样本数量生成标签无关排列、固定batch16，避免原ID顺序可能按类别聚集，真实ID分批私有落盘。预测不是独立逐样本模型，不能按诊断标签搜分批/seed；审计精确复用分批。输入全90帧maskones。原softmaxargmax预测，验证/held1两文件全部冻结后才索引held1标签。原strongestsameheld1参考98.666667%至少+.5pp(1e−12容差)才允许单独source确认；否则close_exact_configuration，无seed/epoch/FFT/宽度/批次救援。探索性八用户历史开发，不称八折/独立验证/原创贡献。

五项真实入口/前任务就绪/审计拒绝活动失败/labelblind分批/末批/门槛守卫通过；三项后处理守卫通过。完整8820步成本估计7386.213932秒已通过43200CPU/wall，独立真实训练43100wall/43000–43100CPU；完整逐步审计14400预算，重放所有训练更新/验证epoch/选模/stopper以及6900RMS与2200最终logits/预测和初始/首步/最优/最后model/RAdam/全部RNG。后处理只等已有训练自然成功→审计一次→本地完整包，不重启训练，无Drive/Git能力，后续需agent私有raw取回校验/重组/全成员SHA再发布最终gate。

协议 research/timesnet_pilot_20261030/PROTOCOL.json SHA bc6885deaf3c33d409cd7a67a694178dee5b8ac4f7f07c95a84991918d6d0349，42项科学/作者源码/前任务证据已冻结。预检期间无真实TimesNet值/标签/拟合，run01不存在。必须先远端逐字读回再一次 python launch.py --run，核实际first_update模型/53组RAdam/16ID后再写真实PID；首步保险包与完整包范围分开。当前记录时真实训练未启动，不声称后台训练。Paper主文OpenReview本轮再次emptycontent/isErrortrue，无详细原因；作者README/代码/脚本为实际一手依据，全文/公式/图未获取，不虚构全文复现，不绕过network策略。旧源码789快照未动，真实数组/模型/实际ID不入Git。目标未达，自动接续保留。

