# DrCIF分块策略仍RSS停止并关闭；完整预训练ResNet18迁移基线实现及成本预登记

截至 2026-11-02T18:03:02.200022+00:00，DrCIF500/CIT20/float64分块频谱screen（登记64a166ff3e5658d89bc05ed49893c0ccbf931cb7，controller89980/worker89981）已在16.554265165003017秒墙钟、20.080412秒CPU后触及12GiBRSS停止线，峰值12636572KiB；监督器TERM−15，cgroupOOMdelta0，full4700fit未完成，没有COST_RESULT、完整模型、准确率或12小时估计。原生整批screen此前也在12.925秒/RSS12737228KiB停止。两次完整规模尝试和修复均保留，原冻结代码、测试、原始失败日志不改不重跑；分块方案的数值一致性检查仍有效，但并未解决完整流水线内存问题。新增只读终止核验已通过。关闭这个确切分块执行配置，不再以树/区间/属性/CIT/精度/seed/视角或进一步同配置资源救援重试，转下个有文献支持的方法。不能把资源失败说成准确率失败或整个家族被否定。

下一项固定为完整官方ImageNet ResNet18冻结编码器及固定线性SVM迁移基线。文献依据分开：Cho/Park2022《Classification of human body motions using an ultra-wideband pulse radar》，TechnologyandHealthCare30(1):93–104，DOI10.3233/THC-212827，出版社主文摘要/3.1–3.3及Table1已读，支持UWB距离时间图像与预训练ResNet18/224×224迁移；该文是人体动作和微距离图像微调，不是本手势数据或冻结SVM方案的直接复现。冻结CNN特征+线性SVM依据Razavian等CVPRWorkshops2014，DOI10.1109/CVPRW.2014.131，原摘要/方法摘录已核查，本文使用现代512维ResNet而非其4096维模型。毫米波手势迁移补充文献DOI10.3390/s21010259已核对题录和主源索引，PMC挑战/MDPI429阻断完整直接读取，不声称作者完整参数已复现。预训练迁移都是已发表技术；本轮只是强基线候选，不能称原创贡献或模块组合创新。

官方pytorch/vision v0.24.0固定commit7a9db90e9aadea3e3e1fec35969601167e9efd4f，resnet.py Gitblob47067ec83175a97cc6f6a8721b342128a434d440。环境没有torchvision二进制包，直接使用官方原样conv3x3/conv1x1/BasicBlock/Bottleneck/ResNet类的AST，数学/结构不改，仅API使用日志为no-op。完整[2,2,2,2]18层网络11689512参数，严格载入全部1000类原始权重后，去掉1000类fc用于512维池化特征；编码器11176512参数全部冻结、BN eval不更新。不是旧1D从头训练ResNet、小改写版或旧冻结LTR特征。

官方默认ImageNet1K_V1权重46830571字节，经允许网络下载并核验发布者SHA前缀及完整SHA=f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec，配置/哈希与公开固定URL保留，权重不进Git。实际源与BSD许可保留。首次源码物化多了末尾换行，被Gitblob断言在模型构造前拦截；已按精确UTF8内容修复，坏副本/失败日志/修复记录保留，未更改作者算法或选用其它权重。

项目输入适配在真实数据前固定：3部雷达按现有顺序作3通道，保持90时间×189距离全图，每条样本/每雷达minmax（floor1e-8），全图双线性缩放224×224、align_cornersFalse、antialiasFalse，不裁剪/增强；ImageNet均值(.485,.456,.406)/标准差(.229,.224,.225)。它不是自然RGB，也不同于ImageNet默认先256后中心裁剪，以避免丢弃半数距离范围；这是明确披露的数据适配，不是新机制或原雷达文作者微距离预处理。未来真实输入仍用既定canonical有符号RMS映射，不能看标签换视角/归一化。

唯一分类头LinearSVC：C1/L2/squared_hinge/dualFalse/tol1e-4/max_iter10000/interceptTrue/scaling1/no classweight/seed20261029。要求收敛，不扫C/权重/骨干/图像尺度/预处理/读出/seed或角色。CPU4、float32、固定batch16、确定性算法；各角色固定顺序分块，不把不同batch大小下的浮点一致性视为已证明。

合成具体检查通过：2条[3,90,189]图像的minmax/全图双线性/ImageNet规范化对独立FP64参考（事先rtol=atol2e-5）通过；完整官方1000类网络原始forward与独立functional卷积残差执行路径logits bitexact；全部冻结参数/BN状态不变；相同分块重复特征和pickle恢复全部状态/特征精确；72条512维/12类合成特征上C1 SVM收敛并恢复系数/截距/预测精确。不是UWB准确率证据，也未做作者微调实验。

新成本目录 `/workspace/uwb-resnet18-imagenet-cost-20261102`，源目录 `/workspace/uwb-resnet18-imagenet-source-20261102`。冻结11文件/3511依赖，COST_HASHES SHA=18de6a0c075828dcf287992c1690d80989867b145e5564101ca92cb7a894b3ed；4项成本边界测试和缺失远端标记阻断检查通过。当前尚未正式计时或启动真实UWB训练。只有全部本次变更精确远端回读及分支头核验后，才能一次完整4700条合成3×90×189输入，经全部ResNet18编码器和C1分类头拟合。逐条可寻址fixture避免整输入占内存，仍保留全部4700条，不做小N外推；全fit计入模型加载、像素适配、全部特征提取和SVM。

资源策略仍3600秒墙钟、CPU14380/14400秒、RSS12GiB、.2秒和cgroupOOM监督。保守3fullfit+2×2200×maxeval64/64+4maxheadsave+2maxheadload+180≤12h，4倍CPU≤48h，峰值+2GiB≤14GiB。冻结编码器为共享权重，2份完整编码器+8个头+64MiB≤剩余磁盘，不能错误要求每个头重复一套冻结编码器，也不能把仅头备份冒称完整模型。真实完整私有归档必须含编码器一次、各头/初始状态、实际ID、预测、日志和源码。成本三次完整encoder+SVM eval64，三次绑定编码器SHA的头保存/恢复，对系数/决策值/预测精确核验。

下一接续先查已有新目录LAUNCH/PID/STATE/TERM，不重复启动。只有正常exit0且无资源停止/OOM才独立verify_result.py一次；资源通过也必须另行冻结登记真实producer、完整数值auditor、私有归档和一次finalizer，不能提前访问实际训练值。真实协议固定train3–6/4700、val2/1000、diag1/1200及canonicalIDs；编码器冻结，只有3–6标签拟合SVM；val支持0..9、diag0..11，固定参考.9866666666666667/+ .005/tol1e-12；完整独立拟合/数值审计及私有全成员恢复后才一次评分，失败关闭确切配置，通过也只source6另行确认。

本轮尚无新UWB准确率、八折提升或原创贡献验证。旧Arsenal96.75%失败及私有992成员恢复记录保持，QUANT一小时未完成关闭保持。用户7–8标签未用于新选择，历史八用户反复开发均探索性。目标未达，原自动接续启用，789原源码、用户README清理、仓库当前可见性及Drive私有性保持；已有数据路径齐全未重下，全部云端不依赖Windows。
