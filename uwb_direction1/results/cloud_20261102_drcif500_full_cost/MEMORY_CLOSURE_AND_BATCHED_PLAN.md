# DrCIF整批频谱内存停止已核验；保持输出的分块执行单独预登记

截至 2026-11-02T17:49:04.418276+00:00，原生DrCIF500/CIT20/float64整批频谱成本screen（登记c8872803679ad80449dd62b862d95d5edcb39c80，controller89167/worker89168）在12.925227639032528秒墙钟、12.911614秒CPU后触及RSS停止线，监督器发送TERM，exit−15。峰值12737228KiB>12GiB，cgroupOOMdelta0，full4700fit未完成，无COST_RESULT、准确率或完整12小时估计。原冻结11文件、2671依赖、科学代码、源码、测试和日志保持；独立只读终止核验通过，旧screen禁止重启。停止发生在整批变换早期，与原生全量float64补零/complexFFT大临时数组的内存需求相符；不是UWB准确率负结果，也未证明整个方法家族不可行。

已具体实现一个执行内存修复，另设目录，不改旧科学配置：`BatchedNativePeriodogram`继承原生PeriodogramTransformer，只预分配完整输出并逐64条调用未修改的原生super()._transform，按原顺序写回。初次新适配器检查发现原生publicfit会重置构造状态、丢弃factory中直接替换的transformer，检查在真实拟合前明确失败，源码/test/log与EXECUTION_REPAIR均保留。现以单独DrCIF执行子类在_fit入口、reset之后安装分块transformer，再委托未修改原生forest；强制核验实际拟合后的transformer类型。这个生命周期修复也不改变森林数学。原float64精度、90补零到128、完整complexFFT后取64bin、567通道/3雷达、500CIT20、10-of29、请求[79,78,67]区间、seed20261029、threading4/BLAS1/Numba1、NaN分支处理、概率平均全部保留。未换rfft、未降低精度、未缩小特征/树/区间/视角、未逐块拟合；只是原生频谱执行分块。现代aeon与论文Java差异仍披露，不能称新算法/原创贡献或原Java精确运行复现。

新增执行一致性检查通过，并非重跑旧冻结测试：原生整批与64条分块频谱在[131,3,90]及[67,567,90]（含常量通道）全元素bitexact，显式跨越64边界；[24,3,90]上两种执行的完整500棵CIT20的全部区间、特征子集、节点、叶概率精确一致，概率、预测和pickle恢复学习状态均精确。完整森林fixture为24条只用单块，跨块正确性由频谱组件的131/67条检查独立覆盖；不是所有可能输入证明，也不是full4700成本通过。旧float32原生Catch22编译故障和输入float64提升修复仍保留，不声称原float32执行可用。原生NaN统计语义保留，未冒称全原始特征有限。

新成本目录 `/workspace/uwb-drcif500-batched-spectrum-cost-20261102`，源适配器目录 `/workspace/uwb-drcif500-batched-spectrum-source-20261102`。冻结11文件/2683依赖，COST_HASHES SHA=80bad024c60b047f54eda892aff0918318dc55da7b5b903a780cf5f29c792cce；成本边界和缺失远端标记阻断检查通过。当前新正式计时尚未启动，真实UWB训练未启动。只有本次所有变更精确回读且分支头核验后，才能在新目录一次full4700×567×90合成fit，保持full500CIT20/allintervals及float64，只把频谱调用分块。不得重启旧89168或改写旧冻结文件/日志，不基于已看诊断标签调执行参数。

新screen仍为3600秒墙钟/CPU软14380硬14400秒/12GiBRSS/.2秒监测/cgroupOOM，无自动重试。完整fit、三次eval64/save/load测时；全部学习状态流式摘要及恢复预测/概率精确。Protocol5非存储sink先计完整模型字节/SHA，8模型+64MiB超空间则显式存储关闭不写检查点；可弃合成副本仅在记录SHA/大小并恢复核验后移除。保守3fullfit+2×2200×maxeval64/64+4maxsave+2maxload+180≤12h，4倍CPU≤48h，实测峰值+2GiB≤14GiB，8模型+64MiB≤剩余磁盘。新策略失败关闭这个确切分块执行配置并转下一项有依据方法，不进行树/核/区间/属性/CIT/seed/精度/视角或标签救援。

下一接续先查已有新目录LAUNCH/PID/STATE/TERM，不重复启动；只有exit0、无停止/OOM时运行独立verify_result.py一次。资源通过后仍需另行登记实际runner、完整独立审计、私有保存和once-finalizer。未来source train3–6/4700、val2/1000、diag1/1200/canonicalIDs，训练标签只用于拟合，不用OOB调参；已知val0..9/diag0..11支持集宏准确率检查，固定参考.9866666666666667/+ .005/tol1e-12；全部审计、私有分块/全成员恢复后才一次评分，失败关闭确切准确率配置，通过也仅source6另行确认。

QUANT完整539217/200树的一小时未完成关闭、Arsenal真实96.75%未过参考及完整私有992成员恢复证明保持。当前没有新UWB准确率、八折提升、独立验证或原创贡献证明。用户7–8标签未参与新选择；历史八用户反复开发均探索性。目标未达，原自动接续启用；789源码、用户README、仓库当前可见性和Drive私有性保持。全部云端，不依赖Windows，已有数据未重下。

备用文献线索已保留为 research/cloud_20261102_future_image_transfer/LEADS.json：毫米波手势ImageNet/ResNet18迁移及UWB人体动作预训练CNN两项主源索引。直接PMC正文遇浏览器挑战，未声称完整读文或作者参数验证；尚未选择、下载权重、实现或拟合。旧1D ResNet缩小实验不自动等于完整预训练2D编码器。当前仍优先完成DrCIF；只有当前配置关闭后才核查备用主文与原代码，不依据已看诊断标签选模型/读出。
