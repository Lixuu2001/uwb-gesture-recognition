# 完整ResNet18微调真实失败关闭；完整ShuffleNetV2 x1固定配置成本预登记

截至2026-11-03T01:53:14.261960+00:00，完整ResNet18全层40epochs/B15/SGDM3e-4/.9/decay1e-4/seed20261029的一次性诊断完成：用户1宏准确率96.9166666667%，验证用户2按固定协议选第22轮macro98.0%。诊断固定参考98.6666666667%，差−1.75个百分点，未达+0.5pp门槛，确切配置关闭，不加轮数/换seed/增强/视角救回，不重评分，不进入source6。原冻结ResNet18+C1的76.6667%失败仍保持关闭。历史八用户反复开发全部探索性；用户7–8标签未用于新选择。当前无新的同协议八折改善证据、无原创贡献证据，目标未达、自动接续继续。

完整审计已实际正常结束：40轮/12560更新逐step损失/顺序/增强RNG、所有40轮验证scores/selection、best及final模型/优化器/动量/随机状态、全部2200 selected输出精确相同；6900 raw RMS/FP64及1038643200基础像素解析float32界全部通过。audit墙钟9174.525354467秒/CPU36335.604221/峰值RSS3648432KiB，exit0/noOOM；producer墙钟10454.170534445/CPU41467.2435。完整流程约5.453小时，低于登记12小时预算。独立进程仍用同源Torch内核，不是第二套卷积实现；旧严格固定2e-5失败与后续解析合同修正历史继续保留，当前模型自始采用登记解析界。审计及producer均已结束，当前没有后台训练或审计运行。

诊断评分之前，完整模型/优化器/随机状态/源码/实际ID/全部日志/所有验证及selected预测归档已恢复验证：171824254字节、24块、146成员（145个payload），archive SHA=14cd8ce9ceac54c78731ace4d99ec7d148444357a9a532a8ceaf4bb98ab1acf0，receipt SHA=afb5888d69904e87a38f0a4682fae253c2025b39809c3e1ef269a239000eb5b4。每块owner-only/private/parent/size元数据先核验，rawtrue/base64false立即物化SHA，最多3条完整流水线；用确认过的远端字节原子替换相同本地传输块，再有序流式校验整体压缩SHA及全部146成员。完整私有模型集通过后才finalize_once一次；排他锁和SUMMARY已存在，不能重复调用。持久Drive索引见同目录complete_backup/CLOUD_DOWNLOAD_INDEX.json，原始数据和运行库另行恢复，不称全操作系统恢复。

之后为下一实验释放空间：仅删除已完整私有恢复且逐块再次SHA确认的24个传输块171824254字节；原始best模型44808971字节、final完整模型/优化器89574031字节、初始12头、全部数据/源码/日志/receipt/索引均保留并核验SHA，云端对象未动。TRANSPORT_RECLAIM.json完整记录路径/字节/SHA/receipt与前后空闲空间。再次恢复这些归档须按持久索引从私有Drive取原始块，禁止重新训练当前失败模型或要求手工拖文件。

下一方法是完整官方ShuffleNetV2 x1.0预训练全层迁移基线。已扫描当前四权威记录，未发现其确切完整native x1/SGDM40试验；扫描有界，非穷尽历史证明。Zhang/Lan/Zhang2021《On the Effect of Training Convolution Neural Network for Millimeter-Wave Radar-Based Hand Gesture Recognition》，DOI10.3390/s21010259、PMID33401744：出版社/PMC可索引的原始摘要、方法摘录及Table3支持预训练ShuffleNetV2用于77GHz手势跨人识别。直接MDPI429、PMC浏览器验证、EuropePMCXML与公开PDF的web读取失败已留痕；不声称完整训练段落已核查。原架构Ma等2018 arXiv1807.11164以及官方torchvision v0.24.0 source commit7a9db90e9aadea3e3e1fec35969601167e9efd4f可复核。

本实现固定完整官方x1.0：[4,8,4]个阶段块、[24,116,232,464,1024]通道，原1000分类2278604参数，换12头后1265904参数全部训练。原雷达论文Table3的1.37M与本canonical x1的2.278604M规模不同；这里不称作者雷达模型/训练严格复现，也不因论文不可立即确认为原创而停止基线实验。三雷达RMS range-time输入不同于77GHz TSV谱图；40完整epochs/B15/SGDM3e-4/.9/decay1e-4/CPU4/f32/seed20261029及共享仿射增强，是项目固定迁移协议（前次Cho/Park2022参数核查支持其预算选择），不冒充Zhang原文超参。架构先验与实际准确率证据分开，普通已发表骨干迁移不称原创。

该唯一配置在本次ResNet18诊断评分前已固定：prospective SHA58a90579655a8188a063f827a1497d41fb7d248bdb13266a487a8064bd2fbbf1，mtime1793670257192972442早于当前一次评分mtime1793670485965333165；当前确切配置失败后才正式选择，无宽度/seed/学习率网格。source blob96736f6a7ac289102ef7a57cb4cbb960c02c625e按Git blob逐字节核验；原channel_shuffle/InvertedResidual/ShuffleNetV2 AST无算法改写，API logging仅no-op。官方ImageNetV1 checkpoint9218294字节，SHA5666bf0f80d1e7f8fb61f6a65742b9c27c37f3208c1f087cdcd7c1ec1aa8b848；HEAD200/元数据先核验，不上传权重Git。

完整native合成检查通过：原1000类logits与独立channel索引重排/分支driver精确一致；B15完整网络、BN、模型/优化器全部动量、增强/Torch随机状态保存恢复后连续2步权重/损失/状态精确一致，完整checkpoint10351235字节（该项IO仅RAM）。新方法尚未读取实际UWB值或标签拟合，正式CPU成本未启动。成本8个文件冻结，COST_HASHES SHA=e3401c37523879a6d10ae83de359c0c5b86d4b8b1f64caae6abfca968de6c435，依赖3529文件含完整源码/权重/运行库已锁定。全部本次变更精确远端回读/head确认后才能启动一次合成cost；不把该提交称为后台新训练已启动。

正式cost固定6个完整B15更新、1个末B5更新、3次B15评估和3次完整模型/优化器/RNG真实磁盘save/load精确校验；公式覆盖两遍40×314更新、两遍40×67验证、selected2200推理及checkpoint/180秒余量，12h/172800CPU/14GiB含2GiB映射余量，筛查上限3600墙钟/14400硬CPU/12GiB RSS。存储包含best模型、final完整优化器、原子写临时、完整初始encoder、全流式private chunks及3个7MiB下载/64MiB余量；保留全部原模型。未过资源门槛关闭确切配置，不缩网络/轮数救回；通过也须另行冻结/预登记真实训练、完整独立重放、私有全成员恢复，再一次诊断。仍固定训练3–6、验证2最高macro/最早并列、诊断1门槛及用户7–8封存；789源码/用户README/仓库当前可见性/Drive私有性保持。
